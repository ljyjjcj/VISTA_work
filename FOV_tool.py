import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import argparse

##This script is used to display the effective search area for the AHEAD system. The script makes a
# number of conservative assumptions about the performance of the algorithm and the camera that are 
# supported by our field tests. 

# --Justin Peel, Arete
#   jpeel@arete.com

# Released 3-1-2022


def derive_ranges(elevation_angle):
    #so, this function will determine the radius at which we could detect
    #a person standing, lying down towards the look, and lying down perpendicular 
    #to the look
    standing = [person_height_m,person_shoulder_m,person_depth_m]
    towards = [person_depth_m,person_shoulder_m,person_height_m]
    perp = [person_shoulder_m,person_height_m,person_depth_m]
    person = np.stack([standing,towards,perp],axis=1)
    el = np.pi*elevation_angle/180
    rot_mat = np.array([[np.cos(el), np.sin(el),0],
                        [-np.sin(el),np.cos(el),0],
                        [0          ,0         ,1]])
    person = rot_mat@person
    max_person_extent = np.max(person[:2,:],axis=0)
    max_distance = max_person_extent*focal_len/(min_pix_height*pix_pitch)
    return max_distance #standing, towards, perp

def make_keystones(elevation_angle, altitude):
    #makes the keystone corners, assuming a pinhole camera. 
    center_distance = altitude/np.tan(elevation_angle*np.pi/180)
    top_bot_ang = elevation_angle+np.array([FOV[1]/2, -FOV[1]/2])
    top_bot_dist = altitude/np.tan(top_bot_ang*np.pi/180)
    corners_x = top_bot_dist*np.cos(FOV[0]/2)
    corners_y = top_bot_dist*np.sin(FOV[0]/2)
    corners = np.array([[corners_x[0],-corners_y[0]],
                        [corners_x[1],-corners_y[1]],
                        [corners_x[1],corners_y[1]],
                        [corners_x[0],corners_y[0]],
                        [corners_x[0],-corners_y[0]]])
    return corners, top_bot_dist



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-a','--altitude', default=20, type=float,
                        help='Platform altitude')
    parser.add_argument('-e','--elevation_angle', default=45, type=float,
                        help='Platform elevation angle in degrees')
    parser.add_argument('-d','--ds_factor', default=4, type=float,
                        help='Downsample factor')

    args = parser.parse_args()

    #base camera specs. These are for the IMX 274, the current sensor on the AHEAD system. These
    # should be updated if we change the sensor or the lens. 
    pix = np.array([3864,2196])
    focal_len = 0.005 # in meters, so I don't need to convert later.
    pix_pitch = 1.62e-6 # meters, again
    sensor_size = pix_pitch*pix #officially a [5.76,4.29] 1/2.5 format, but this works better for a nonlinear lens
    FOV = 2*np.arctan(pix_pitch*pix/(2*focal_len))*180/np.pi # officially [76,43], but this calculation underestimates by a bit
    F_num = 2.2


    #detection rules of thumb
    min_pix_height = 32*args.ds_factor
    person_height_m = 1.75 #about 5'7"
    person_shoulder_m = 0.4 #futzed numbers for a kinda smaller person
    person_depth_m = 0.2

    standing, towards, perp = derive_ranges(args.elevation_angle)
    corners, radii = make_keystones(args.elevation_angle, args.altitude)
    print(f"Detection range for standing {standing:3.1f}m, laying towards {towards:3.1f}m, and laying perpendicular {perp:3.1f}")
    if standing>radii[1]: standing=1.01*radii[1]
    if towards>radii[1]: towards=1.02*radii[1]
    if perp>radii[1]: perp=1.03*radii[1]


    ang = np.sin(FOV[0]/2)*180/np.pi
    arc_top = mpatch.Arc([0,0],2*radii[0],2*radii[0],0,90-ang,90+ang)
    arc_bot = mpatch.Arc([0,0],2*radii[1],2*radii[1],0,90-ang,90+ang, label='_nolegend_')
    arc_stand = mpatch.Arc([0,0],2*standing,2*standing,0,90-ang,90+ang,color='red')
    arc_perp = mpatch.Arc([0,0],2*perp,2*perp,0,90-ang,90+ang,color='green')
    arc_towards = mpatch.Arc([0,0],2*towards,2*towards,0,90-ang,90+ang,color='blue')

    fig,ax = plt.subplots(1,1); 
    plt.title(f"Elevation Angle: {args.elevation_angle}, Altitude: {args.altitude}, Downsample: {args.ds_factor}")
    ax.add_patch(arc_top); 
    ax.add_patch(arc_bot); 
    ax.add_patch(arc_stand); 
    ax.add_patch(arc_perp); 
    ax.add_patch(arc_towards); 
    ax.scatter(corners[:,1],corners[:,0],color='k', label='_nolegend_'); 
    ax.plot(corners[0:2,1],corners[0:2,0],'k', label='_nolegend_')
    ax.plot(corners[2:4,1],corners[2:4,0],'k', label='_nolegend_')
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=corners[1,1],right=corners[2,1])
    ax.legend(['FOV','standing','laying right','laying towards'])
    plt.show()