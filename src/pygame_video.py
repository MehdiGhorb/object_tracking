import pygame
import sys
import cv2
from utils.path import *

# Set screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 30
VIDEO__LENGTH = 20
OUTPUT_VIDEO_PATH = os.path.join(ORIGINAL_VIDEOS_DIR, "moving_star.mp4")

# Set colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# List of possible shapes
# 'circle', 'square', 'star', etc.
SHAPES = ['circle', 'square', 'star']

class Shape:
    def __init__(self, screen, shape):
        self.screen = screen
        self.shape = shape
        self.x = SCREEN_WIDTH // 2
        self.y = SCREEN_HEIGHT // 2
        self.speed_x = 5
        self.speed_y = 5

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
            pygame.draw.circle(self.screen, BLACK, (self.x, self.y), 15)
        elif self.shape == 'square':
            pygame.draw.rect(self.screen, BLACK, (self.x - 15, self.y - 15, 30, 30))
        elif self.shape == 'star':
            # Draw a star (customize as needed)
            points = [(self.x, self.y - 20), (self.x + 10, self.y + 5), (self.x + 25, self.y + 10),
                      (self.x + 15, self.y + 25), (self.x + 20, self.y + 40), (self.x, self.y + 30),
                      (self.x - 20, self.y + 40), (self.x - 15, self.y + 25), (self.x - 25, self.y + 10),
                      (self.x - 10, self.y + 5)]
            pygame.draw.polygon(self.screen, BLACK, points)

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

    shape = Shape(screen, 'star')  # Default shape is a circle

    frames = []
    total_frames = FPS * VIDEO__LENGTH

    # Main loop
    for frame_count in range(total_frames):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        screen.fill(WHITE)

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
