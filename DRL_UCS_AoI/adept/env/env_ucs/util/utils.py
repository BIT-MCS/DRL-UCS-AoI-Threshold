# task热力图可视化
import seaborn as sns
#import imageio
import matplotlib.pyplot as plt
import numpy as np 
sns.set_theme(style="white")




def cross(p1,p2,p3):
    x1=p2[0]-p1[0]
    y1=p2[1]-p1[1]
    x2=p3[0]-p1[0]
    y2=p3[1]-p1[1]
    return x1*y2-x2*y1     

def IsIntersec(p1,p2,p3,p4): 

    if(max(p1[0],p2[0])>=min(p3[0],p4[0])    
    and max(p3[0],p4[0])>=min(p1[0],p2[0]) 
    and max(p1[1],p2[1])>=min(p3[1],p4[1])  
    and max(p3[1],p4[1])>=min(p1[1],p2[1])): 

        if(cross(p1,p2,p3)*cross(p1,p2,p4)<=0
           and cross(p3,p4,p1)*cross(p3,p4,p2)<=0):
            return True
        else:
            return False
    else:
        return False


if __name__=='__main__':
    pass