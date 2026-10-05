### OCR model

**Process large batch of text images as a txt file**

**To run model on your machine** 


>Note it requires Python3.13+
**For mac/linux**
```python
!python3 -m venv .venv
source .venv/bin/activate
```
**For windows**
```python
!python -m venv .venv
.venv\Scripts\activate.bat
```
**Install required libraries**
```python
!pip install requirements.txt
```
>Note file path for model.pt needs to be correct in the .env 
 
 After installing the requirements 
 run the cv_inf.py 

 > The cv_inf.py opens a camera window , direct image batch processing hasnt been built yet, the inference code is only to see the model work 

 