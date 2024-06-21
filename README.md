# Text-to-Music Retrieval++ 

This is a implementation of [TTMR++](#)(Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieval). This project aims to search music with text query. 
<!-- 
> [**Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieval**](#)

> SeungHeon Doh, Minhee Lee, Dasaem Jeong and Juhan Nam 
> IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) 2024 
 -->

# Installation

1. Environment Preparation 
(Please install torch according to your [CUDA version](https://pytorch.org/get-started/previous-versions/))
    ```python
    python==3.8
    torch==1.12.1 (Please install it according to your CUDA version.)
    ```
2. Other requirements
    ```python
    pip install -e .
    ```

# Usage

### 0. Preparation
- Downloaded [Annotation mp3 files] (annotationfilelink) should be in `../Data/Annotation/mp3`

- Download the [Pre-trained model weights](https://huggingface.co/seungheondoh/ttmr-pp/tree/main) and put them into `../Data/`


### 1. Data Preprocess
Process your mp3 files into npy files. (You can skip the process when you are using processed npy files.)
```python
cd preprocessing
python mp3_to_npy_preprocessor.py
```

### 2. Embedding Extraction 
```python
python extractor.py
```
Extract embeddings of the dataset with pretrained model(ttmr++). 

After run this code, **two embedding files**(audio/tag) will be saved in `../exp/annotation/embs/` with certain directory setting in the code. 

Extracted embedding files can also be downloaded [Embedding files](http://embedding) 

### 3. Train Probing Layer
```python
python train_probing.py
```
After training, model_best.pth file will be saved in the same folder with embedding files. 

### 4. Evaluation
```python
python eval.py
```
