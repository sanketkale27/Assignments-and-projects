Following are the steps to run the chatbot
1.	set the all file path 
a.	DATA_PATH :  dataset path
b.	OUTPUT_FILE : outputfile
c.	PROCESSED_PATH : set the path where you want to store the train and test data or else it will create on the default file path
d.	CPT_PATH :  set the path where the checkpoints will stroe or else it will create on the default file path

•	If you want to train the dataset then follow the following streps 

1.	Run the data.py file and create the train and test data
2.	Chatbot.py file – in the main function change the argument as  default = ‘train’(line number 381)
3.	After train the model change the argument as default = “chat” and run the chatbot
4.	Ouput_convo file maintains the chat history 
5.	Feedback file maintains the chat which user wants to update
6.	Joey file contains the basic information of joey
7.	Detailed Files folder contains the section wise files which is needed for the execution



