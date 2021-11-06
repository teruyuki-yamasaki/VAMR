# Exercie 01 - Augmented reality wireframe cube


### Step 0 : Load dataset 
- K.txt : the camera matrix K
- D.txt : the parameters for the camera's radial distortion model
- poses.txt : the camera's orientation and position for each frame given as a 6D vector
```
def txt2array(txt, sep=' '):
    return np.fromstring(txt, dtype=float, sep=sep)

def load_data():
    K = open("./data/K.txt").read()
    K = txt2array(K).reshape(3,3) 

    D = open("./data/D.txt").read() 
    D = txt2array(D) 

    poses = open("./data/poses.txt").read()
    P = txt2array(poses).reshape(len(poses.split('\n')[:-1]), 6)

    filenames = glob.glob('./data/images/*.jpg')   
    filenames = sorted(filenames) 

    return K, D, P, filenames
```

### Step 1 : Calculate the Transformation Matrix
