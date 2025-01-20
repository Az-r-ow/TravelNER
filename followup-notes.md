# Follow up notes

## Follow up 1

**Date:** Friday 11 October 2024

### Work done

- Gradio Interface
- Folder Structure
- Shortest Path Algorithm
- V1 of the NER model
- Language Recognition

### Questions

- Any recommendations on vectorizing the sentences ?
- Should we go for character or word based vectorization ?

### Answers

- A good start would be with word based vectorization it's simpler and involves less dimensions

### Remarks

Our idea to train and compare different models seems interesting to Prof. Nassar and he would like to see an F1-score comparison chart.

## Follow up 2

**Date:** Friday 29 November 2024

### Work done

- HMM
- Trained and evaluated LSTM
- Trained and evaluated BiLTM
- Starting to train BERT

### Remarks

Prof. Nassar said to use `cmarkea/distilcamembert-base` instead of `camembert-base` because it converges faster.

## Follow up 3

**Date:** Friday 17 January 2025

### Work done

- Trained and evaluated CamemBERT
- Tested and evaluated LSTM and BiLSTM with POS as extra features
- Modified the interface to include a selected for the model
- Refactored the interface and added tabs for files with multiple sentences

### Remarks

Prof. Nassar was happy with our work and excited to see the final product on the keynote.
