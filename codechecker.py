/**
 * @author  Jagadeesh Thiruveedula
 * @version 3.7.6
 * @Lang    Python
 * Distribution Anaconda
 */

#this is basically a file checker program
file = open('/mlops/mnist.py','r')	
#post loading file into file varibale it will read 
checker = file.read()				
# this is crucical part 

if 'keras' or 'tensorflow' in checker:			
	if 'Conv2D' in code:				
		print('CNN')
	else:
		print('not CNN')
else:
	print('not deep learning')
