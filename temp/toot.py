import numpy as np


x = [np.array([[ 0.00226779],
       [-0.00721124],
       [-0.00956809],
       [-0.00799385],
       [-0.00411143],
       [-0.01118377],
       [-0.00891052],
       [-0.00695755],
       [-0.0039394 ],
       [-0.00232481],
       [-0.00346364],
       [-0.01264513],
       [-0.01471389],
       [-0.0136413 ],
       [-0.01338116]], dtype=np.float32), np.array([[ 0.00095623],
       [-0.00670712],
       [-0.01072183],
       [-0.00637447],
       [-0.004015  ],
       [-0.01053523],
       [-0.01011444],
       [-0.00861351],
       [-0.00567403],
       [-0.00379257],
       [-0.00534201],
       [-0.01144282],
       [-0.01921221],
       [-0.02109047],
       [-0.0158196 ]], dtype=np.float32), np.array([[-0.01028183],
       [-0.00633332],
       [-0.01674626],
       [-0.01168632],
       [-0.00794425],
       [-0.00654582],
       [-0.01006559],
       [-0.0046736 ],
       [-0.01028666],
       [-0.00866444],
       [-0.0121335 ],
       [-0.0121401 ],
       [-0.0082102 ],
       [-0.01292782],
       [-0.01385576]], dtype=np.float32), np.array([[-0.00806321],
       [-0.004454  ],
       [-0.01658622],
       [-0.01408   ],
       [-0.00496898],
       [-0.00866814],
       [-0.00991621],
       [-0.0070807 ],
       [-0.01203341],
       [-0.01155963],
       [-0.01279465],
       [-0.01136759],
       [-0.00587528],
       [-0.01094869],
       [-0.01146293]], dtype=np.float32), np.array([[-0.00700711],
       [-0.00916655],
       [-0.01208138],
       [-0.01400401],
       [-0.00624665],
       [-0.00855223],
       [-0.00845524],
       [-0.00204677],
       [-0.01452512],
       [-0.00965587],
       [-0.01350115],
       [-0.01049942],
       [-0.00718795],
       [-0.0096594 ],
       [-0.01666512]], dtype=np.float32)]

x2 = [np.array([[5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5]]), np.array([[5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5]]), np.array([[5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5]]), np.array([[5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5]]), np.array([[5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5],
       [5]])]

for y in x2:
    print(y.shape)