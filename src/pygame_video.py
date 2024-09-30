import pygame
import sys
import cv2
import os
import random

# Set screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 30
VIDEO_LENGTH = 5
OUTPUT_VIDEO_PATH = '../assets/train_videos_shapes/moving_star.mp4'

# Set colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)
BLUE = (0, 0, 255)
PINK = (255, 192, 203)

object_color = BLUE
background_color = PINK # Change the background color here
sh = 'star'

class Shape:
    def __init__(self, screen, shape):
        self.screen = screen
        self.shape = shape
        # Initialize position randomly within the screen boundaries
        self.x = random.randint(0, SCREEN_WIDTH)
        self.y = random.randint(0, SCREEN_HEIGHT)
        # Initialize direction randomly
        self.speed_x = random.choice([-5, 5])
        self.speed_y = random.choice([-5, 5])

    def update(self):
        self.x += self.speed_x
        self.y += self.speed_y

        # Bounce back when hitting the walls
        if self.x >= SCREEN_WIDTH or self.x <= 0:
            self.speed_x *= -1
        if self.y >= SCREEN_HEIGHT or self.y <= 0:
            self.speed_y *= -1

    def draw(self):
        if self.shape == 'circle':
            pygame.draw.circle(self.screen, object_color, (self.x, self.y), 4)
        elif self.shape == 'rectangle':
            pygame.draw.rect(self.screen, object_color, (self.x - 15, self.y - 15, 30, 30))
        elif self.shape == 'star':
            # Draw a star (customize as needed)
            points = [(self.x, self.y - 20), (self.x + 10, self.y + 5), (self.x + 25, self.y + 10),
                      (self.x + 15, self.y + 25), (self.x + 20, self.y + 40), (self.x, self.y + 30),
                      (self.x - 20, self.y + 40), (self.x - 15, self.y + 25), (self.x - 25, self.y + 10),
                      (self.x - 10, self.y + 5)]
            pygame.draw.polygon(self.screen, object_color, points)
        elif self.shape == 'triangle':
            pygame.draw.polygon(self.screen, object_color, [(self.x, self.y - 15), (self.x + 15, self.y + 15), (self.x - 15, self.y + 15)])

def save_video(frames, fps, output_path):
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

def main():
    pygame.init()
    clock = pygame.time.Clock()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Moving Object")

    shape = Shape(screen, sh)  # Default shape is a circle

    frames = []
    total_frames = FPS * VIDEO_LENGTH

    # Main loop
    for frame_count in range(total_frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Fill the screen with background color
        screen.fill(background_color)

        shape.update()
        shape.draw()

        pygame.display.flip()

        # Capture the screen as an image
        image = pygame.surfarray.array3d(screen)
        image = cv2.transpose(image)
        image = cv2.flip(image, 0)
        frames.append(image)

        clock.tick(FPS)

    # Save the captured frames as a video
    save_video(frames, FPS, OUTPUT_VIDEO_PATH)

    pygame.quit()

if __name__ == "__main__":
    main()
