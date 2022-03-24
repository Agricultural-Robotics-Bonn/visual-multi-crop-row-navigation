class Camera:
    def __init__(self, id,deltax,deltay,deltaz,tilt_angle,height,width,focal_length,sensor_width):
        self.id = id
        self.deltax = deltax                   # Translation to the origin of the robot
        self.deltay = deltay
        self.deltaz = deltaz                    # Heigth of the camera
        self.tilt_angle = tilt_angle            # Tilt Angle
        self.height = height                    # Camera Intrinsics
        self.width = width
        self.focal_length = focal_length
        self.sensor_width = sensor_width