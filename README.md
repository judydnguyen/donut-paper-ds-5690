<div align="center">
    
# Donut 🍩 : Document Understanding Transformer

Donut and SynthDoG | [Paper](https://arxiv.org/abs/2111.15664) | [Poster](https://docs.google.com/presentation/d/1m1f8BbAm5vxPcqynn_MbFfmQAlHQIR5G72-hQUFS2sk/edit?usp=sharing) | [Code](https://github.com/clovaai/donut)

</div>

## Authors
Geewook Kim, Teakgyu Hong, Moonbin Yim, Jeongyeon Nam, Jinyoung Park, Jinyeong Yim6, Wonseok Hwang, Sangdoo Yun, Dongyoon Han, Seunghyun Park

## Introduction
**Donut** 🍩, **Do**cume**n**t **u**nderstanding **t**ransformer, is a new method of document understanding that utilizes an OCR-free end-to-end Transformer model. Donut does not require off-the-shelf OCR engines/APIs, yet it shows state-of-the-art performances on various visual document understanding tasks, such as visual document classification or information extraction (a.k.a. document parsing). 
In addition, we present **SynthDoG** 🐶, **Synth**etic **Do**cument **G**enerator, that helps the model pre-training to be flexible on various languages and domains.

## Overview

### What problem does this article solve?
Donut addresses the following problems:
- Dependency on OCR engines: Most existing document understanding models depend heavily on external OCR systems, which are prone to errors, limited to specific languages, and costly for large-scale data processing.
- Complex multi-stage pipelines: OCR-based pipelines involve multiple steps like detection, recognition, and classification, increasing processing time and error propagation.
- Limited support for multi-lingual documents: Traditional systems struggle to perform equally well across different languages and scripts without re-training or custom datasets.
  
### What are the shortcomings of existing solutions?
- Accuracy degradation due to OCR errors: OCR can introduce recognition errors, especially in noisy or complex layouts. These errors propagate through the entire VDU pipeline.
- Limited domain adaptability: Most OCR models are trained for general-purpose text and perform poorly on specialized documents such as forms, invoices, or handwriting.
- High computational cost: Multi-stage VDU pipelines require high computation, making real-time applications challenging.

### Solution
Donut offers an end-to-end Transformer-based architecture that operates directly on images, eliminating the need for OCR. Its key innovations include:
- Image-to-text generation for document parsing.
- Multi-task training to support a variety of VDU tasks, from classification to extraction.
- SynthDoG for language-agnostic pre-training, improving generalization across diverse datasets and languages.
- Efficiency: With no intermediate OCR processing, Donut delivers faster inference and simpler deployment.

## Architecture Overview
Donut’s architecture is built on Transformer-based sequence modeling principles. Unlike traditional models, which first convert images into text via OCR, Donut treats the entire document image as input to a Vision Transformer (ViT) and processes it directly.

Encoder-decoder structure: The encoder handles the document image, extracting features, while the decoder generates the required output, such as extracted text or document labels.
Image-to-sequence modeling: Donut outputs sequences that represent text spans or key-value pairs directly from the input image.
Multi-task learning: It can perform various tasks, such as classification, extraction, summarization, and parsing, by fine-tuning on corresponding datasets.

### In document understanding task that is not Donut-based, what is SOTA's approach to do it?
The current SOTA approach for document understanding relies on OCR-based pipelines like LayoutLM and Tesseract. These pipelines follow these steps:

- Detect text regions using object detection.
- Recognize text within these regions via OCR.
- Classify or extract key information from recognized text.

While effective, these systems are prone to cascading errors (OCR errors affect later steps) and struggle with multi-lingual content. Donut offers a simpler, unified solution by avoiding these intermediate steps altogether.

### Introduce Donut in detail
Donut’s key features include:
- OCR-free document understanding: Bypasses the need for text recognition, working directly with images.
- Efficient pre-training with SynthDoG: Generates synthetic data to train the model, allowing it to perform well on unseen domains and low-resource languages.
- Supports multiple tasks: Can classify documents, extract key information, and summarize content with a single model.
Better generalization: Learns from synthetic multi-lingual data for improved robustness across languages and layouts.

### Experiments
- The authors evaluated Donut on several standard VDU benchmarks, including RVL-CDIP (classification) and CORD (information extraction). The authors conducted experiments with three down-stream tasks: (i) document classification, (ii) document parsing and (iii) document VQA.

- The results showed:
SOTA performance on classification and extraction tasks without using OCR engines.
Faster inference compared to traditional pipelines, demonstrating its potential for real-time applications.
Experiments with SynthDoG highlighted Donut’s ability to adapt to new domains without fine-tuning, confirming the generalizability of the pre-trained model.

## Critical Analysis

**Strengths:**

- OCR-free design: Eliminates common bottlenecks in VDU pipelines.
- Simplicity: A unified end-to-end model for various tasks.
- Multi-lingual and domain-adaptive: Performs well across diverse datasets and languages.

**Limitations:**

- *Training Data Bias:* Pre-training on synthetic data alone may not generalize well to real-world scenarios without further domain-specific fine-tuning (4 languages)
- *High Pre-training Cost:* Training Donut with 11M images and large-scale synthetic data requires significant computational resources (e.g., 64 A100 GPUs), making it less accessible for smaller research teams or organizations with limited resources.
- *Challenging for deployments:* Although Donut achieves faster inference than OCR-based systems, it still relies on heavy transformer architectures that might not be suitable for low-power or mobile devices.
- *OOD performance concern:* Donut may struggle with tiny or low-contrast text in documents where OCR-based systems with image pre-processing steps could perform better. For example, under OOD layouts, font-sizes, languages, etc.


## Questions for Discussion
1. What do you think will be the factors affecting the performance of training Donut?
2. Why was the Swin Transformer selected as the visual encoder, and could alternative backbones (e.g., ViT) provide better results?
3. In which cases where Donut fails to extract accurate information (e.g., as shown in the DocVQA examples), what are the key reasons? Would additional fine-tuning with specialized datasets solve these issues?

## Demonstrating of using Donut
- Implementation of fine-tuning Donut [here](Fine_tune_Donut_on_DocVQA.ipynb)
- Implementation of using Donut [here](demo.ipynb)

## Additional Resources

- [Link to Paper](https://arxiv.org/abs/2111.15664)
- [PapersWithCode link](https://paperswithcode.com/paper/donut-document-understanding-transformer)
- [Fine-tuning Donut model tutorial](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Donut/DocVQA/Fine_tune_Donut_on_DocVQA.ipynb)
- [Huggingface Tutorial](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa)
- [Blog of A Comprehensive Analysis](https://ubiai.tools/fine-tuning-donut-model-on-docvqa-a-comprehensive-analysis/)


## Citation
```
@inproceedings{kim2022donut,
  title     = {OCR-Free Document Understanding Transformer},
  author    = {Kim, Geewook and Hong, Teakgyu and Yim, Moonbin and Nam, JeongYeon and Park, Jinyoung and Yim, Jinyeong and Hwang, Wonseok and Yun, Sangdoo and Han, Dongyoon and Park, Seunghyun},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022}
}
```