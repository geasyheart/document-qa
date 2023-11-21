import base64
import io
import random
from collections import Counter
from typing import List, Optional, Union
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from PIL.ImageDraw import ImageDraw
from datasets import load_dataset
from torch.nn import BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers.modeling_outputs import QuestionAnsweringModelOutput

raw_dataset = load_dataset('./docvqa_zh')

train_dataset = raw_dataset['train']
dev_dataset = raw_dataset['validation']
test_dataset = raw_dataset['test']


# train_dataset = train_dataset.select(range(300))
# dev_dataset = dev_dataset.select(range(10))

def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def preprocess(example):
    assert example['bbox'] == example['segment_bbox']
    text = example['text']
    qas = example['qas']
    new_qas = {}
    for question_id, question, answer in zip(qas['question_id'], qas['question'], qas['answers']):
        # 1. 没答案/找不到也要
        new_qas.setdefault(question, set())
        for answer_start, answer_end, answer_text in zip(answer['answer_start'], answer['answer_end'], answer['text']):
            if answer_start == -1 and answer_end == -1: continue
            if "".join(text[answer_start:answer_end - 1]) != answer_text:
                # 总共query：64323 有效query：61242
                # print("".join(text[answer_start:answer_end - 1]), answer_text)
                continue
            else:
                new_qas.setdefault(question, set()).add((answer_start, answer_end - 1))
    # bbox = example['bbox'][710:716]
    # print("".join(text[710:716]))
    # img = Image.open(io.BytesIO(base64.b64decode(example['image'])))
    # draw = ImageDraw.ImageDraw(img)
    # for b in bbox:
    #     draw.rectangle(b, outline='red')
    # img.show()
    img = Image.open(io.BytesIO(base64.b64decode(example['image'])))
    if img.mode != "RGB":
        img = img.convert('RGB')
    result = []
    for q, a in new_qas.items():
        # 2. 判断答案相同
        # assert len(set(["".join(text[_si:_ei]) for (_si, _ei) in a])) <= 1

        # 这里添加一步，比如
        # 图片介绍了什么活动？
        # 答案是：京东女神节 每满400减40
        # 但是这两个不在同一行
        # 所以这里进行找最多的那个
        if a:
            target_answer = Counter(["".join(text[_si:_ei]) for (_si, _ei) in a]).most_common(1)[0][0]
            new_a = [(_s, _e) for (_s, _e) in a if "".join(text[_s:_e]) == target_answer]
        else:
            new_a = list(a)
        result.append({
            "name": example['name'],
            "image": img,
            "question": q,
            "answers": new_a,
            "words": text,
            "boxes": example['bbox']
        })
    return result


def _show(example):
    image = example['image']
    draw = ImageDraw(image)
    for (si, ei) in example['answers']:
        for box in example['boxes'][si:ei]:
            draw.rectangle(box, outline=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255),))
    print(example['question'])
    image.show()


from transformers import LayoutXLMProcessor, LayoutLMv2ForQuestionAnswering

checkpoint_path = '/data/.tmpyz/layoutxlm-base/'

processor = LayoutXLMProcessor.from_pretrained(checkpoint_path, apply_ocr=False)
image_processor = processor.image_processor
tokenizer = processor.tokenizer


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


def process(example):
    image_width, image_height = example['image'].size
    example['boxes'] = [normalize_box(box, width=image_width, height=image_height) for box in example['boxes']]

    pixel_values = image_processor(example['image']).pixel_values[0]

    res = _tokenizer(text1=example['question'], text2=example['words'], bbox=example['boxes'])
    res.update({"pixel_values": pixel_values})

    # 计算标签
    start_index = res['text1_len']
    if not example['answers']:
        res.update({"start_positions": [0], "end_positions": [0]})
        return res
    else:
        start_positions, end_positions = [], []
        for (word_idx_start, word_idx_end) in example['answers']:
            start_position = word_idx_start + start_index
            end_position = word_idx_end + start_index
            _convert_answer = res['input_ids'][start_position:end_position]
            _origin_answer = tokenizer.convert_tokens_to_ids(example['words'][word_idx_start:word_idx_end])

            assert _convert_answer == _origin_answer
            start_positions.append(start_position)
            end_positions.append(end_position)
        res.update({"start_positions": start_positions, "end_positions": end_positions})
        return res


class DocVQADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


trains, devs, tests = [], [], []

for data in train_dataset:
    for example in preprocess(example=data):
        # origin_res = processor(
        #     example['image'],
        #     # text=" ".join([_ for _ in example['question']]),
        #     text=" ".join(tokenizer.tokenize(example['question'])),
        #     text_pair=example['words'],
        #     boxes=example['boxes'],
        #     max_length=512, padding="max_length", truncation=True
        # )

        data = process(example)

        if len(data['input_ids']) > 512:
            # 这玩意未来能解决
            continue
        trains.append(data)
print(f"total train dataset length:{len(trains)}")
for data in dev_dataset:
    for example in preprocess(example=data):
        data = process(example)
        if len(data['input_ids']) > 512:
            continue
        devs.append(data)
print(f"total dev dataset length:{len(devs)}")
from transformers import Trainer, TrainingArguments


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


training_args = TrainingArguments(
    output_dir='./output',
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    evaluation_strategy='epoch',
    num_train_epochs=50,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_total_limit=1,
    load_best_model_at_end=True,
    save_strategy='epoch',
    metric_for_best_model='f1',
    remove_unused_columns=False,
    dataloader_pin_memory=False,
    gradient_accumulation_steps=2

)


def collate_fn(examples):
    input_ids, attention_mask, bbox, image = [], [], [], []
    start_positions, end_positions = [], []
    for example in examples:
        input_ids.append(torch.tensor(example['input_ids'], dtype=torch.long))
        attention_mask.append(torch.tensor(example['attention_mask'], dtype=torch.long))
        bbox.append(torch.tensor(example['bbox'], dtype=torch.long))
        image.append(example['pixel_values'])
        sp = torch.zeros(len(example['input_ids']))
        ep = torch.zeros(len(example['input_ids']))
        for si in example['start_positions']:
            sp[si] = 1
        for ei in example['end_positions']:
            ep[ei] = 1
        start_positions.append(sp)
        end_positions.append(ep)

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    bbox = pad_sequence(bbox, batch_first=True, padding_value=0)
    start_positions = pad_sequence(start_positions, batch_first=True, padding_value=0)
    end_positions = pad_sequence(end_positions, batch_first=True, padding_value=0)
    image = torch.from_numpy(np.stack(image)).to(torch.float)
    return {
        "input_ids": input_ids.to("cuda"),
        "bbox": bbox.to("cuda"),
        "image": image.to("cuda"),
        "attention_mask": attention_mask.to("cuda"),
        "start_positions": start_positions.to("cuda"),
        "end_positions": end_positions.to("cuda")
    }


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(e):
    from umetrics.macrometrics import MacroMetrics
    start_macro = MacroMetrics([0, 1])
    end_macro = MacroMetrics([0, 1])
    true_start_indices = e.label_ids[0][e.label_ids[0] != -100].tolist()
    true_end_indices = (e.label_ids[1][e.label_ids[1] != -100]).tolist()
    pred_start_indices = [1 if _ > 0.5 else 0 for _ in sigmoid(e.predictions[0][e.label_ids[0] != -100])]
    pred_end_indices = [1 if _ > 0.5 else 0 for _ in sigmoid(e.predictions[1][e.label_ids[1] != -100])]
    start_macro.step(y_trues=true_start_indices, y_preds=pred_start_indices)
    end_macro.step(y_trues=true_end_indices, y_preds=pred_end_indices)

    start_f1 = start_macro.f1_score()
    end_f1 = end_macro.f1_score()
    print(f"start_f1: {start_f1}, end_f1: {end_f1}")
    return {"f1": (start_f1 + end_f1) / 2}


model = MultiLayoutLMv2ForQuestionAnswering.from_pretrained(checkpoint_path)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=DocVQADataset(trains),
    eval_dataset=DocVQADataset(devs),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train('/data/.tmpyz/document-qa/output/checkpoint-78720/')
