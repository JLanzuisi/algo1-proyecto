#!/usr/bin/env python
# coding: utf-8

# # Segunda entrega
# Esta debería ser la entrega final.
# 
# A continuación están las funciones.

# In[1]:


def read_img(image: str) -> [list,int,int,float]:
    from skimage import io
    import numpy as np
    import cv2

    img = io.imread(image)

    X = img.shape[0]
    Y = img.shape[1]
    pixsum = np.sum(img)

    imgread = [img,X,Y,pixsum]
    
    return imgread

# Test
# print(read_img("./Freedo_improved.jpeg")[1])
# print(read_img("./Freedo_improved.jpeg")[2])
# print(read_img("./Freedo_improved.jpeg")[3])


# In[2]:


def draw_triangles(X,Y,triangles):
    import numpy as np
    import cv2

    image = np.ones((X,Y,3), np.uint8)*255 # Blank squared image

    N = len(triangles)

    for i in range(0,N):
        pts =  np.array(
                [triangles[i][0],triangles[i][1],triangles[i][2]],
                np.int32
               )
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(image, [pts], (triangles[i][3]))

    return image


# In[3]:


def first_gen(N: int, P: int, X: int, Y: int) -> [list,list]:
    import random
        
    triangles = []
    imagearr = []
    for i in range(0,P):
        vertex = []
        for i in range(0,N):
            triag = [
            	[random.randint(0,X),random.randint(0,Y)],
            	[random.randint(0,X),random.randint(0,Y)],
                [random.randint(0,X),random.randint(0,Y)],
                [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
            ]
            vertex += [triag]
        imagearr += [draw_triangles(X,Y,vertex)]
        triangles += [vertex]
    
    return [imagearr,triangles]

# Test
#first_gen(5,5,204,209)[0][0]


# In[4]:


def fitness(original, image):
    import numpy as np
    import cv2

    X = read_img("./Freedo_improved.jpeg")[1]
    Y = read_img("./Freedo_improved.jpeg")[2]
    
    difflist = []
    for i in range(0,X):
        for k in range(0,Y):
            origvals = []
            imgvals = []
            for l in range(0,3):
                origvals += [original.item(i,k,l)]
                imgvals += [image.item(i,k,l)]
            if origvals[0] == imgvals[0] and origvals[1] == imgvals[1] and origvals[2] == imgvals[2]:
                difflist += [1]
            else:
                difflist += [0]
                
    diffsum = sum(difflist)
    diff = sum(difflist)/len(difflist)
    
    return diff
# Test
#fitness(read_img("./Freedo_improved.jpeg")[0],read_img("./Freedo_improved_inverted.jpeg")[0])


# In[5]:


def mutate(triangles,X,Y):
    import random

    N = len(triangles)
    
    M = random.randint(0,N-1)
    K = random.randint(0,3)

    colorpercent = int(255 * 0.05)
    Xvertpercent = int(X * 0.05)
    Yvertpercent = int(Y * 0.05)

    for i in range (0,M):
        l = random.randint(0,N-1)
        for i in range (0,K):
            k = random.randint(0,3)
            if k != 3:
                triangles[l][k][0] += random.randint(1,Xvertpercent)*random.choice([-1,1])
                triangles[l][k][1] += random.randint(1,Yvertpercent)*random.choice([-1,1])
                if triangles[l][k][0] >= X:
                    triangles[l][k][0] -= X
                if triangles[l][k][1] >= Y:
                    triangles[l][k][1] -= Y
            else:
                triangles[l][k][0] += random.randint(1,colorpercent)*random.choice([-1,1])
                triangles[l][k][1] += random.randint(1,colorpercent)*random.choice([-1,1])
                triangles[l][k][2] += random.randint(1,colorpercent)*random.choice([-1,1])
                if triangles[l][k][0] >= 255:
                    triangles[l][k][0] -= 255
                if triangles[l][k][1] >= 255:
                    triangles[l][k][1] -= 255
                if triangles[l][k][2] >= 255:
                    triangles[l][k][1] -= 255

    return triangles

# Test
# triangles = first_gen(5,5,204,209)[1][2]
# print(triangles)
# mutate(triangles,204,209)


# In[74]:


def selection(original,imagearr) -> [int]:
    import random
    
    N = len(imagearr)

    difflist = []
    for i in range(0,N):
        difflist += [[fitness(original,imagearr[i]),i]]

    difflist = sorted(difflist, reverse=True)

    selected = []
    selected += [difflist[0][1]]

    diffsum = 0
    problist = []
    for i in range(1,N):
        diffsum += difflist[i][0]
        problist += [[diffsum,difflist[i][1]]]

    
    M = int(N * 0.9)
    for i in range(0,M):
        end = N-1
        start = 0
        r = random.uniform(0,diffsum)
        while end != start+1:
            mid = (end+start)//2
            if r > problist[mid][0]:
                start = mid
            elif r < problist[mid][0]:
                end = mid
            else:
                end = start+1
        selected += [problist[end][1]]
    
    # https://www.w3schools.com/python/python_howto_remove_duplicates.asp
    selected = list(dict.fromkeys(selected))

    return selected

# Test
#selection(read_img("./Freedo_improved.jpeg")[0],first_gen(12,10,204,209)[0])


# In[75]:


def crossover(parentA:list, parentB:list, X: int, Y: int,):
    import random
    
    N = len(parentA)

    sonA = []
    sonB = []

    for i in range(0,N):
        if i <= N//2:
            sonA += [parentA[i]]
            sonB += [parentB[i]]
        else:
            sonB += [parentA[i]]
            sonA += [parentB[i]]

    if random.uniform(0,100) <= 5:
        if random.randint(0,1) == 1:
            sonA = mutate(sonA,X,Y)
        else:
            sonB = mutate(sonB,X,Y)

    return [sonA,sonB]

# Test
#crossover(first_gen(15,7,204,209)[1][0],first_gen(15,7,204,209)[1][1],204,209)


# In[ ]:


def next_gen(original,imagearr,triangles):
    import cv2

    img = original
    X = img.shape[0]
    Y = img.shape[1]
    
    selected = selection(original,imagearr)
    
    N = len(imagearr)

    nextgentriag = []
    for i in range(0,len(selected)):
        nextgentriag += [triangles[selected[i]]]

    i = 1
    while len(nextgentriag) < N:
        sons = crossover(nextgentriag[i],nextgentriag[i+1],X,Y)
        nextgentriag += [sons[0]]
        nextgentriag += [sons[1]]
        i += 1

    while N != len(nextgentriag):
        nextgentriag = nextgentriag[:-1]

    nextgenimgarr = []
    
    for i in range(0,N):
        nextgenimgarr += [draw_triangles(X,Y,nextgentriag[i])]

    return [nextgenimgarr,nextgentriag]
    
# Test
#next_gen(read_img("./Freedo_improved.jpeg")[0],first_gen(6,10,204,209)[0],first_gen(6,10,204,209)[1])[1]


# In[9]:


def gen_algo(original: str, N: int, P: int,):
    from matplotlib import pyplot as plt
    import cv2
    
    X = read_img(original)[1]
    Y = read_img(original)[2]
    img = read_img(original)[0]

    firstgen = first_gen(N,P,X,Y)

    nextgen = next_gen(img,firstgen[0],firstgen[1])
    
    print(fitness(img,nextgen[0][0]))

    difflist = []
    L = 100000
    for i in range(0,L):
        nextgen = next_gen(img,nextgen[0],nextgen[1])
        difflist += [fitness(img,nextgen[0][0])]   
        
    #for i in range(0,P):
        #difflist += [fitness(img,nextgen[0][i])]

    plt.plot(difflist)
    plt.show()

    #cv2.imshow("window2",nextgen[0][0])
    #cv2.waitKey(0)
    #cv2.destroyWindow('window2')
    #cv2.waitKey(1)

    #return nextgen

# Test
gen_algo("./simple-rectangle.jpg",20,20)


# In[ ]:




