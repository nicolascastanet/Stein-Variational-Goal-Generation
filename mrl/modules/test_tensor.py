import numpy as np

def tensor_point_maze(density=0.4):        
    h = density
    x_min, x_max = -2.5, 11.5
    y_min, y_max = -2.5, 11.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    goal_test_tensor = np.c_[xx.ravel(), yy.ravel()]

    

    return goal_test_tensor


def tensor_ant_maze(density=0.4):
    h = density

    # up
    x_min, x_max = -4, 20
    y_min, y_max = 12, 20
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    up = np.c_[xx.ravel(), yy.ravel()]

    # middle
    x_min, x_max = 12, 20
    y_min, y_max = 4, 12
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    mid = np.c_[xx.ravel(), yy.ravel()]

    # down
    x_min, x_max = -4, 20
    y_min, y_max = -4, 4
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

    down = np.c_[xx.ravel(), yy.ravel()]

    goal_test_tensor = np.concatenate((up,mid,down))

    return goal_test_tensor