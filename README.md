## 说明

本项目是基于layoutxlm在docvqa_zh上一次尝试。

在计算loss那里，没有使用官方的实现，因为官方代码假设一个图片中只会有一个答案，而在docvqa_zh中，会有多个，以及不一致的情况。

## 可优化方向

1. 蛮多文本是超过512个字符的，可以尝试使用layoutxlm-large来做实现。
2. 使用更大的batch_size。