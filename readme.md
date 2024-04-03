##SAC v1.4
**state：**(np.array([self.distance,velo_angle_rad,x_dis,y_dis,z_dis]),self.blue_vel)  
>x_dis, y_dis, z_dis represent the closest distance to the boundary on each direction, velo_angle represents the angle between velo vector and position vector   

**action：**[ax,ay,az]  
training target：stay alive and chased a target at [9.9,9.9,9.9], target radius is 0.1  
**done:** get out of the zone (fail) or reach target (success)  
**reward：** 0.1+delta_distance or 300 if success 
result as follow, training completed at 14600th episode
![](2024-01-16-12-11-07.png)  
**note:** use gymnasium.spaces with different boundary at different dimensions to output time interval and acceleration simutaneously
**analysis:** state space has more dimension (7 to 6) than v1.2, so the training is significantly slower; but fewer state dimensions, say 5, would be insufficient to describe the system, so it's important to keep a physically minimal state dimension while redesigning the state space