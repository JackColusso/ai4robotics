Modifications:
* Added third-person and on-board camera rendering modes.
* Made runnable on google colab notebook

# Gym-Medium-Post
Basic OpenAI gym environment. 

Resource for the [Medium series on creating OpenAI Gym Environments with PyBullet](https://medium.com/@gerardmaggiolino/creating-openai-gym-environments-with-pybullet-part-1-13895a622b24). 

DQN Training model has been implimented in assignment3.py. Takes approx 15mins to run
Training data set has been saved into train_dqn_model.pth
For quick testing, use the run.py code to load the trained model and see if car reaches goals while avoiding obstacles. This will save time so you dont have to run the training model
Obstacles are placed between the goal and origin each run
If car hits obstacle, terminal will print a message saying there was a collision 
DO NOT remove or replace the trained_dqn_model. it has been trained multiple times using the previous data set to improve on the current one
