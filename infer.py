import os
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from torch.nn import BCEWithLogitsLoss
from transformers import LayoutXLMProcessor, LayoutLMv2ForQuestionAnswering
from transformers.modeling_outputs import QuestionAnsweringModelOutput


def do_ocr(filepath) -> List[Dict]:
    with open(filepath, 'rb') as f:
        result = requests.post('http://localhost:5000', files={'file': (os.path.basename(filepath), f)}).json()
        return result


def _tokenizer(text1: str, text2: List[str], bbox: List):
    """
    注意：一定要看下processor的实现哦
    比如：默认实现每个中文前面添加_作为开始，我这里忽略掉了
    :param text1:
    :param text2:
    :return:
    """
    # <s>token1</s></s>token2</s>
    text1 = tokenizer.tokenize(text1)
    input_tokens = ['<s>', *text1, '</s>', '</s>', *text2, '</s>']
    attention_mask = [1] * len(input_tokens)

    # <s>
    new_bbox = [[0, 0, 0, 0]]

    for _t1_token in text1:
        new_bbox.append([0, 0, 0, 0])

    new_bbox.append([1000, 1000, 1000, 1000])

    new_bbox.append([1000, 1000, 1000, 1000])

    for _i, _t2_token in enumerate(text2):
        new_bbox.append(bbox[_i])
    new_bbox.append([1000, 1000, 1000, 1000])

    return {
        "tokens": input_tokens,
        "input_ids": tokenizer.convert_tokens_to_ids(input_tokens),
        "attention_mask": attention_mask,
        'bbox': new_bbox,
        "text1_len": 1 + len(text1) + 2
    }


checkpoint_path = "document-qa/output"

processor = LayoutXLMProcessor.from_pretrained(checkpoint_path, apply_ocr=False)
image_processor = processor.image_processor
tokenizer = processor.tokenizer


class MultiLayoutLMv2ForQuestionAnswering(LayoutLMv2ForQuestionAnswering):
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            bbox: Optional[torch.LongTensor] = None,
            image: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            start_positions: Optional[torch.LongTensor] = None,
            end_positions: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.layoutlmv2(
            input_ids=input_ids,
            bbox=bbox,
            image=image,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        # only take the text part of the output representations
        sequence_output = outputs[0][:, :seq_length]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = BCEWithLogitsLoss()
            mask = attention_mask.bool()
            start_loss = loss_fct(start_logits[mask], start_positions[mask])
            end_loss = loss_fct(end_logits[mask], end_positions[mask])
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


model = MultiLayoutLMv2ForQuestionAnswering.from_pretrained(checkpoint_path)


def infer(query: str, image_path: str):
    pil_img = Image.open(image_path)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert('RGB')
    image_width, image_height = pil_img.size

    pixel_values = image_processor(pil_img).pixel_values[0]

    res = do_ocr(image_path)
    tokens, boxes = [], []
    for _ in res:
        for char in _['text']:
            tokens.append(char)
            boxes.append(normalize_box(_['bbox'], width=image_width, height=image_height))
    res = _tokenizer(text1=query, text2=tokens, bbox=boxes)

    pred = model.forward(input_ids=torch.LongTensor(res['input_ids']).reshape(1, -1),
                         bbox=torch.LongTensor(res['bbox']).reshape(1, len(res['bbox']), 4),
                         image=torch.from_numpy(pixel_values.copy().reshape(1, 3, 224, 224)).float(),
                         attention_mask=torch.LongTensor(res['attention_mask']).reshape(1, -1))

    start_logits = sigmoid(pred.start_logits.detach().numpy())[0]
    end_logits = sigmoid(pred.end_logits.detach().numpy())[0]

    # si = np.argwhere(start_logits > 0.5)
    # ei = np.argwhere(end_logits > 0.5)
    # print(si, ei)
    sis, eis = [], []
    for si, prob in enumerate(start_logits):
        if prob > 0.5:
            sis.append(si)
    for ei, prob in enumerate(end_logits):
        if prob > 0.5:
            eis.append(ei)

    spans = []
    for si in sis:
        for ei in eis:
            if ei > si:
                spans.append((si, ei))
                break

    for (si, ei) in spans:
        print(tokenizer.decode(res['input_ids'][si:ei]))


if __name__ == '__main__':
    infer('标题是什么？', './check_ocr/img-0.png')
