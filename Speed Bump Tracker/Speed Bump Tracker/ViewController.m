//
//  ViewController.m
//  Speed Bump Tracker
//
//  Created by Andrew Zolintakis on 9/25/17.
//  Copyright © 2017 Andrew Zolintakis. All rights reserved.
//

#import "ViewController.h"
#import <CoreMotion/CoreMotion.h>
#import <CoreLocation/CoreLocation.h>

@interface ViewController (){
    CMMotionManager *motionManager;
    CLLocationManager *locationManager;
    NSString* filePath;
    NSString *writeString;
    int speedBump;
    int potHole;
    double oldx;
    double oldy;
    double oldz;
    
}

@end

@implementation ViewController
static double const g_force_conversion = 9.80665;


- (void)viewDidLoad {
    [super viewDidLoad];
    motionManager = [[CMMotionManager alloc] init];
    motionManager.accelerometerUpdateInterval = 0.25;
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    filePath = [NSString stringWithFormat:@"%@/%@", documentsDirectory, @"accelerometer-data.csv"];
    
    locationManager = [[CLLocationManager alloc] init];
    locationManager.delegate = self;
    [locationManager requestAlwaysAuthorization];
    
    
    if (![[NSFileManager defaultManager] fileExistsAtPath:filePath]) {
        [[NSFileManager defaultManager] createFileAtPath: filePath contents:nil attributes:nil];
        NSLog(@"Route creato");
    }
    
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}


- (IBAction)startButtonTouched:(id)sender {
    if([self.startButton.titleLabel.text isEqualToString:@"Start"]){
        if([CLLocationManager authorizationStatus] == kCLAuthorizationStatusDenied || [CLLocationManager authorizationStatus] == kCLAuthorizationStatusRestricted || [CLLocationManager authorizationStatus] == kCLAuthorizationStatusNotDetermined){
             [locationManager requestAlwaysAuthorization];
            return;
        }
        [locationManager startUpdatingLocation];
        writeString= @"longitude, latitude, X, Y, Z, speedbump, pothole\n";
        speedBump = 0;
        potHole = 0;
        oldx = 0;
        oldy = 0;
        oldz = 0;
        [self.startButton setTitle:@"Stop" forState:UIControlStateNormal];
        self.startButton.backgroundColor = [UIColor redColor];
        NSLog(@"lets go");
        [motionManager startAccelerometerUpdatesToQueue:[[NSOperationQueue alloc] init] withHandler:^(CMAccelerometerData *data, NSError *error)
        {
            
            dispatch_async(dispatch_get_main_queue(),
                           ^{
                               
                               double newx = data.acceleration.x / g_force_conversion;
                               double newy = data.acceleration.y/ g_force_conversion;
                               double newz = data.acceleration.z/ g_force_conversion;
                               
                               double xx = newx - oldx;
                               double yy = newy - oldy;
                               double zz = newz - oldz;
                               
                               double longitude = locationManager.location.coordinate.longitude;
                               double latitude = locationManager.location.coordinate.latitude;
                               
                               NSString *dataString = [NSString stringWithFormat:@"%f, %f, %f, %f, %f, %d, %d\n", longitude, latitude, xx, yy, zz, speedBump, potHole];
                               
                               writeString = [writeString stringByAppendingString: dataString];
                               
                               oldx = newx;
                               oldy = newy;
                               oldz = newz;
                               speedBump = 0;
                               potHole = 0;
                           });
        }];
    }else{
        [self.startButton setTitle:@"Start" forState:UIControlStateNormal];
        self.startButton.backgroundColor = [UIColor greenColor];
        [motionManager stopAccelerometerUpdates];
        [writeString writeToFile:filePath atomically:YES encoding:NSUTF8StringEncoding error:nil];
        NSLog(@"Data in File");
    }
}
- (IBAction)touchedSpeedBump:(id)sender {
    speedBump = 1;
}
- (IBAction)touchedPothole:(id)sender {
    potHole = 1;
}
- (NSUInteger) supportedInterfaceOrientations {
    // Return a bitmask of supported orientations. If you need more,
    // use bitwise or (see the commented return).
    return UIInterfaceOrientationMaskPortrait;
    // return UIInterfaceOrientationMaskPortrait | UIInterfaceOrientationMaskPortraitUpsideDown;
}

- (UIInterfaceOrientation) preferredInterfaceOrientationForPresentation {
    // Return the orientation you'd prefer - this is what it launches to. The
    // user can still rotate. You don't have to implement this method, in which
    // case it launches in the current orientation
    return UIInterfaceOrientationPortrait;
}
- (IBAction)share:(id)sender {
    if([[NSFileManager defaultManager] fileExistsAtPath:filePath]){
        UIDocumentInteractionController *documentController;
        documentController = [UIDocumentInteractionController interactionControllerWithURL:[NSURL fileURLWithPath:filePath]];
        documentController.UTI = @"public.csv";
        [documentController presentOpenInMenuFromRect:CGRectZero
                                               inView:self.view
                                             animated:YES];
    }
}



@end
