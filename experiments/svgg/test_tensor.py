import numpy as np

def tensor_point_maze(density=0.4,xy_min=-0.5,xy_max=9.6):        
    h = density
    xx,yy = np.meshgrid(np.arange(xy_min, xy_max, h),np.arange(xy_min, xy_max, h))

    goal_test_tensor = np.c_[xx.ravel(), yy.ravel()]

    return goal_test_tensor

def tensor_point_maze_random(d=11,nb_sample=10):
    x = np.linspace(-0.5, 9.5, d)
    y = np.linspace(-0.5, 9.5, d)
    random_goals = []

    for i in range(len(x)-1):
        for j in range(len(y)-1):
            data = np.random.uniform(low=[x[i],y[j]], high=[x[i+1],y[j+1]], size=(nb_sample,2))
            random_goals.append(data)
    
    return np.array(random_goals).reshape(-1,2)

    



def tensor_maze_2(density=0.4):
    h=density
    # 1 bar
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -10.5, 0.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    bar_1 = np.c_[xx.ravel(), yy.ravel()]

    # 2 bar
    x_min, x_max = 3.5, 4.5
    y_min, y_max = -10.5, 0.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    bar_2 = np.c_[xx.ravel(), yy.ravel()]

    # 2 bar
    x_min, x_max = 7.5, 8.5
    y_min, y_max = -10.5, 0.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    bar_3 = np.c_[xx.ravel(), yy.ravel()]

    #2 middle
    x_min, x_max = 0.5, 3.5
    y_min, y_max = -10.5, -9.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    mid_1 = np.c_[xx.ravel(), yy.ravel()]

    x_min, x_max = 4.5, 7.5
    y_min, y_max = -0.5, 0.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    mid_2 = np.c_[xx.ravel(), yy.ravel()]

    goals = np.concatenate((bar_1,bar_2,bar_3,mid_1,mid_2))

    return goals


def tensor_maze_square_c2(density=0.4):
    h=density
    # 1 bar
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -4.5, 0.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    bar_1 = np.c_[xx.ravel(), yy.ravel()]

    # 2 bar
    x_min, x_max = 1.5, 2.5
    y_min, y_max = -4.5, 0.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    bar_2 = np.c_[xx.ravel(), yy.ravel()]

    # 2 bar
    x_min, x_max = 3.5, 4.5
    y_min, y_max = -4.5, 0.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    bar_3 = np.c_[xx.ravel(), yy.ravel()]

    #2 middle
    x_min, x_max = 0.5, 1.5
    y_min, y_max = -4.5, -3.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    mid_1 = np.c_[xx.ravel(), yy.ravel()]

    x_min, x_max = 2.5, 3.5
    y_min, y_max = -0.5, 0.5
    xx,yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    mid_2 = np.c_[xx.ravel(), yy.ravel()]

    goals = np.concatenate((bar_1,bar_2,bar_3,mid_1,mid_2))

    return goals    

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