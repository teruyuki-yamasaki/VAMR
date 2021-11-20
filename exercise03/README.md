# Harris detector 

To implement Harris corner detection and key point tracking: First evaluate the Harris score for each pixel of the input image, then select key points based on the Harris scores, and finally match descriptors in order to find feature correnspondences between frames. 

<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/data/000000.png"/>

[imgrad.py](https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/code/imgrad.py)
([test](https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/code/test_imgrad.py))

<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/results/imgrad_Ix.png"/>
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/results/imgrad_Iy.png"/>

[scores.py](https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/code/constructStructureTensor.py)

<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/results/shitomashi.png"/>
<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/results/harris.png"/>

#### TODO:
- score のヒストグラムの可視化

[matches.py](https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/code/matches.py)

<img src="https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/results/matches.png"/>

- np.uniqueがおかしなことをしていないか？
- keypoint matching の高速化

[main.py](https://github.com/teruyuki-yamasaki/VAMR/blob/main/exercise03/code/main.py)
