# EC-523-Project
This repository is dedicated to the re-implementation and enhancement of the novel SCalable Remembering and Unlearning unBound (SCRUB) model.


**SCRUB**
```python3
python3 main.py --checkpoints --sgda_learning_rate 0.006 --sub_sample 0.0 --unlearning_method "SCRUB" 
```
**Finetuning**
```python3
python3 main.py --checkpoints --sgda_learning_rate 0.1 --unlearning_method "finetuning" 
```
**Negative gradient**
```python3
python3 main.py --checkpoints --sgda_learning_rate 0.006 --unlearning_method "negative" --sgda_epochs 5
```



