# Static_Sign_language_recognition
This repository contains code for static sign language recognition for ASL data sets.
Models have been trained with incremental modifications.
1.	TF_vr1: Base Network
2. TF_vr2: Learning Rate Reduction Added
3.	TF_vr3: Batch Normalization Added
4.  TF_vr4: Dropout Layers Added
5.  TF_vr5: Size of Dense Layers Increased <- Best performing Model( Both in F1 score and inference timings)
6.  Efficient Net B0

To recreate the experiment please run the test.py file. Please note that inference times have been calculated on jetson nano 4GB.
These readings can vary with the load on the system at the moment. Though an average of the readings have been taken after multiple runs,
but it can still vary depending on the current system state.
Architecture of the final model.


![image](https://github.com/skrmanglam/Static_Sign_language_recognition/assets/31559064/440488e2-0802-41d1-bd35-0b1d562fb203)

