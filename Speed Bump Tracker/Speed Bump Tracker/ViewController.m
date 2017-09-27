//
//  ViewController.m
//  Speed Bump Tracker
//
//  Created by Andrew Zolintakis on 9/25/17.
//  Copyright © 2017 Andrew Zolintakis. All rights reserved.
//

#import "ViewController.h"
#import <CoreMotion/CoreMotion.h>

@interface ViewController (){
    CMMotionManager *motionManager;
    NSString* filePath;
    NSString *writeString;
    NSString *speedBump;
    NSString *potHole;
    
}

@end

@implementation ViewController


- (void)viewDidLoad {
    [super viewDidLoad];
    motionManager = [[CMMotionManager alloc] init];
    motionManager.accelerometerUpdateInterval = 0.25;
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    filePath = [NSString stringWithFormat:@"%@/%@", documentsDirectory, @"accelerometer-data.csv"];
    
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
        writeString= @"X, Y, Z, speedbump, pothole\n";
        speedBump = @"no";
        potHole = @"no";
        [self.startButton setTitle:@"Stop" forState:UIControlStateNormal];
        self.startButton.backgroundColor = [UIColor redColor];
        NSLog(@"lets go");
        [motionManager startAccelerometerUpdatesToQueue:[[NSOperationQueue alloc] init] withHandler:^(CMAccelerometerData *data, NSError *error)
        {
            
            dispatch_async(dispatch_get_main_queue(),
                           ^{
                           
                               double xx = data.acceleration.x;
                               double yy = data.acceleration.y;
                               double zz = data.acceleration.z;
                               
                               NSString *dataString = [NSString stringWithFormat:@"%f, %f, %f, %@, %@\n", xx, yy, zz, speedBump, potHole];
                               
                               writeString = [writeString stringByAppendingString: dataString];
                               speedBump = @"no";
                               potHole = @"no";
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
    speedBump = @"yes";
}
- (IBAction)touchedPothole:(id)sender {
    potHole = @"yes";
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
