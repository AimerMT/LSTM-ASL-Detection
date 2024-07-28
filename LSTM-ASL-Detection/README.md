Truong Vy Hoa
hoa.truongvyhoa118@hcmut.edu.vn

Step 0: installing libraries 

Step 1: Create your dataset with *hands_data_generation.py*
 - Change the label name (line 12), change it to the label of your dataset.
 - Determine how many hands you want to capture the motion (line 8). Replace max_num_hands to 2 if you want to capture both hands

Step 2: Train the LSTM model with *train_model.py*
 - model.add(Dense(units=X, activation="softmax")) - replace X with the number of your dataset
 - After your train, a .h5 file will be created.  

Step 3: Run a demo with your model with *hands_lstm_realtime_custom.py*
 - detect function: select the highest prediction from the list of trained dataset and give the results, you'll need to modify this function to adapt to your application

 
Demo video:
https://www.youtube.com/watch?v=-A4pXOoC5Ns