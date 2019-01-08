from __future__ import division
import sys
import cv2
import numpy as np
import time
import threading
from random import randint
from math import *
from scipy import *
from scipy import linalg as LA
from numpy import linalg as lA
from scipy.spatial.distance import cdist
import numpy.matlib
import os, os.path
import scipy.io


def LDA(X, L):
        
        Lx = unique(L)
        Classes = transpose([Lx])
        k = Classes.shape[0]*Classes.shape[1]
        z= X.shape[0]
        n = zeros((k,1))
        S = zeros((z,z))
        C = zeros((z,k))
        M = mean(X, axis=1)
		Sw = 0
        Sb = 0
        for j in range(0,k):
                #var = (L == Classes[j])
                #var = var.T
		
                Xj = X[:,(L == Classes[j])]
                Xj = np.array(Xj)
               
                #Xj = transpose([Xjj])
		n[j] = Xj.shape[1]
                C[:,j]= mean(Xj, axis=1)
            	S=0
            	
                for i in range(0,n[j]):  
                		          
                        x = Xj[:,i] - C[:,j]
                        xtt = np.transpose([x])                                                           
                        S = S + xtt*x
			
                Sw = Sw + S
               		
		csbt= np.transpose([(C[:,j] - M)])
		
                Sb = Sb + n[j] * csbt*(C[:,j] - M) 
	
	
	tt3 = np.dot(1/Sw,Sb)
	jw, w_t = LA.eig(tt3);
	jw = np.real(jw);
	w_t = np.real(w_t);
	jw = np.array(jw)
	w_t = np.array(w_t)
	srt = jw.argsort()[::-1][:1]
	W = w_t[:,srt]
	print W
        
        return W

def img_conv(inp):
	
	x=[]
	for i in range(0,inp):
		A11 = cv2.imread('50x50/'+str(i+1)+'.jpg',0);
		A11 = im2double(A11);
		A12 = cv2.resize(A11, (50,50),interpolation=cv2.INTER_CUBIC);
        	z1 = A12.shape[0]*A12.shape[1];
        	A1 = np.reshape(A12,(1,z1));
		A1 = np.transpose(A1)
		if i==0:
			x = column_stack([A1])
		else:
			x = column_stack([x,A1])
	return x
	
def im2double(im):
	min_val = np.min(ravel(im))
	max_val = np.max(ravel(im))
	out = ((im.astype('float')) - min_val) / (max_val - min_val)
	return out



def main():
        X=img_conv(30)
        #print x.shape
        L=np.array([1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3])
      #X = np.array([[4,2,2,3,4,9,6,9,8,10],[2,4,3,6,4,10,8,5,7,8]]);
      #L=np.array([1,1,1,1,1,2,2,2,2,2])
        W= LDA(X,L)
        W = np.transpose(W)
	Y = np.dot(W,X)
	Y = np.transpose(Y)
	Y = np.array(Y)
	print Y	
	

	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	#video_capture = cv2.VideoCapture(0)
	video_capture = cv2.VideoCapture('V6.mp4')
	count  = 0;
	start = time.time();
	while True:	
	    # Capture frame-by-frame
	    count = count + 1
	    ret, frame = video_capture.read()

	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
    	
    		
	    faces = faceCascade.detectMultiScale(
	        gray,
	        scaleFactor=1.9,
	        minNeighbors=2,
	        minSize=(60,60),
	        flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	    )
	    
	   
    # Draw a rectangle around the faces
	    for (x, y, w, h) in faces:
	        
	        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		roi=frame[y:y+h, x:x+w]
	        
        	T11 = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY);
        	T12 = im2double(T11);
		T12 = cv2.resize(T12,(50,50),interpolation = cv2.INTER_CUBIC)
        	zt = T12.shape[0]*T12.shape[1];
        	Tz = np.reshape(T12,(1,zt));
		Tz = np.transpose(Tz)
		
		ans = np.dot(W,Tz)
		ans = np.transpose(ans)
		ans = np.array(ans)
		
		
		z=10
		#y1 = Y[0:z];
       		#y2 = Y[z:2*z];
       		#y3 = Y[2*z:3*Z]
       		
       	#	y1m= mean(y1)
       #		y2m=mean(y2)
       #		y3m=mean(y3)
       		
       		
		CP=[]
        	
		for ii in range(0,3):
        	#	s=cdist(,np.array([ans]),'mahalanobis');#Database_LDA=Trained data projected
			
			s = sqrt(sum( (mean(Y[ii*z:(ii+1)*z]) - ans.T) ** 2))
			#print s
			#print 'Distance :', math.sqrt(s)
			#print ' '
        		if ii==0:	
	        		CP = row_stack([s]);
			else:
				CP = row_stack([CP,s]);
		
		
		
		#CP = cdist(ans,Y,'mahalanobis')
		#print CP
		#print '\n'
        	index_min = CP.argmin() + 1
		#print index_min
		#if index_min<12:
		#	clss = 1
		#elif index_min>11 and index_min<23:
	#		clss = 2
#		else:
#			clss = 3
			
		#if CP[index_min-1]<=W*thr[index_min-1]:
        	#print index_min
        	cv2.putText(roi, str(index_min), (5, 25),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255))
        	#else:
		#	print 'NOT FOUND'
        	
		    # Display the resulting frame
	    cv2.imshow('Video', frame)
	
	    if cv2.waitKey(1) & 0xFF == ord('q'):
	        break
	end = time.time()
	seconds = end - start
	#print "Time taken : {0} seconds".format(seconds)
	#print count
	 
	    # Calculate frames per second
	fps  = count / seconds;
	#print "Estimated frames per second : {0}".format(fps);
# When everything is done, release the capture
	video_capture.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

