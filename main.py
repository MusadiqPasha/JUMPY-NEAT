"""Importing all the needed modules"""

import pygame as pg
from pygame.locals import RLEACCEL
import os
import neat
import time
import random
import networkx as nx
import matplotlib.pyplot as plt

"""# Doodler class

The Doodler class represents a character in a game, This class encapsulates all the functionality and attributes associated with the doodler character.


**Attributes**:

    left_img and right_img: Surface objects representing the doodler's images when facing left and right directions, respectively.

    height and width: Dimensions of the doodler's image.

    prev_pos and pos: Previous and current positions of the doodler.

    vel: Velocity of the doodler.

    acc: Acceleration of the doodler.

    score: Current score of the doodler.

    gravity: Magnitude of gravitational force affecting the doodler.

    facing_right: Boolean indicating whether the doodler is facing right.

    lost: Boolean indicating whether the doodler has lost the game.

    score_line: Vertical position representing the baseline for scoring.

    collision: Boolean indicating whether the doodler has collided with an object.

    start_movement: Boolean indicating whether the doodler has started moving.

**Methods**:

    display(surf): Displays the doodler's image on a given surface.

    land_on_platform(dt, platform): Adjusts the doodler's position and velocity when it lands on a platform.

    move(dt): Updates the doodler's position based on its velocity and acceleration.

    apply_gravity(dt): Applies gravitational force to the doodler.

    update_movement(dt, left, right): Updates the doodler's

    movement based on user input and current state.

    move_right(dt) and move_left(dt): Moves the doodler horizontally to the right or left, respectively.

    is_dead(screen_height): Determines whether the doodler has fallen below a certain position on the screen, indicating that the game is over.

# Code
"""

class Doodler:
    def __init__(self, starting_pos):
        # Load the images for the doodler facing left and right
        self.left_img = pg.image.load(r"images/rightdoodler.png").convert()
        self.left_img.set_colorkey((255, 255, 255), RLEACCEL) # Set white as transparent
        self.right_img = pg.image.load(r"images/leftdoodler.png").convert()
        self.right_img.set_colorkey((255, 255, 255), RLEACCEL) # Set white as transparent

        # Get dimensions of doodler image
        self.height = self.left_img.get_height()
        self.width = self.left_img.get_width()

        # Set initial position, velocity, and acceleration of doodler
        self.prev_pos = starting_pos
        self.pos = starting_pos
        self.vel = (0, 0)
        self.acc = (0, 0)

        # Initialize other doodler attributes
        self.score = 1
        self.gravity = 0.0000020
        self.facing_right = True
        self.lost = False
        self.score_line = 0
        self.collision = False
        self.start_movement = False

    def display(self, surf):
        # Display the appropriate image of the doodler based on its direction
        if self.facing_right:
            surf.blit(self.right_img, self.pos)
        else:
            surf.blit(self.left_img, self.pos)

    def land_on_platform(self, dt, platform):
        # Adjust doodler properties when landing on a platform
        self.vel = (self.vel[0], -0.03*dt*(self.score**0.05)) # Adjust velocity based on score
        self.acc = (self.acc[0], 0) # Set acceleration to zero horizontally
        self.pos = (self.pos[0], platform.pos[1] - self.height) # Position doodler on top of platform
        self.prev_pos = (self.pos[0], platform.pos[1] - self.height) # Update previous position

    def move(self, dt):
        # Move the doodler based on its velocity and acceleration
        if not self.collision:
            self.prev_pos = self.pos
            self.pos = (self.pos[0] + self.vel[0]*dt, self.pos[1] + self.vel[1]*dt)
            self.vel = (self.vel[0] + self.acc[0]*dt, self.vel[1] + self.acc[1]*dt)

    def apply_gravity(self, dt):
        # Apply gravity to the doodler
        self.acc = (self.acc[0], self.acc[1] + self.gravity*dt*(self.score**0.1))

    def update_movement(self, dt, left, right):
        # Update doodler movement based on input
        if left:
            if self.start_movement:
                self.vel = (-.001*dt, self.vel[1])
                self.acc = (0, self.acc[1])
            self.move_left(dt)
            self.start_movement = False
        elif right:
            if self.start_movement:
                self.vel = (.001*dt, self.vel[1])
                self.acc = (0, self.acc[1])
            self.move_right(dt)
            self.start_movement = False
        elif self.vel[0] > 0.0005*dt:
            self.acc = (-0.0001*dt, self.acc[1])
            self.start_movement = True
        elif self.vel[0] < -0.0005*dt:
            self.acc = (0.0001*dt, self.acc[1])
            self.start_movement = True
        else:
            self.vel = (0, self.vel[1])
            self.acc = (0, self.acc[1])
            self.start_movement = True

    def move_right(self, dt):
        # Move doodler to the right
        self.facing_right = True
        self.acc = (.0001*dt, self.acc[1])
        if self.vel[0] > 0.02*dt:
            self.vel = (0.02*dt, self.vel[1])

    def move_left(self, dt):
        # Move doodler to the left
        self.facing_right = False
        self.acc = (-.0001*dt, self.acc[1])
        if self.vel[0] < -0.02*dt:
            self.vel = (-0.02*dt, self.vel[1])

    def is_dead(self, screen_height):
        # Check if doodler has fallen below a certain position on the screen
        return self.pos[1] > self.score_line + 0.66*screen_height + self.height

"""# Platform Class

The Platform class represents a platform within the game environment. It encapsulates functionality related to the platform's appearance, behavior, and interaction with other game elements.


**Attributes:**

    pos: The position of the platform on the game screen.

    img: The image representing the platform's appearance.

    height and width: The dimensions of the platform's image.

    type: The type of the platform, indicating whether it is still or moving.

    vel: The velocity of the platform (applicable for moving platforms).

**Methods:**

    display(surf): Displays the platform's image on the specified surface.

    collided_width(player): Detects collisions between the platform and the player character.

    in_view_of(doodler, screen_height): Checks if the platform is within the view of the doodler character.

    is_too_close_to(other_plat, screen_width): Checks if the platform is too close to another platform.

    move(screen_width): Moves the platform horizontally, with screen wrapping if it reaches the screen edges.

    screen_wrap(screen_width): Handles screen wrapping for the platform.

**Helper Functions:**

    intersect(p1, p2, p3, p4): Determines if two line segments intersect.

    ccw(p1, p2, p3): Determines if three points are in counter-clockwise order.

# Code
"""

class Platform():
    def __init__(self, pos, type):
        # Initialize platform attributes
        self.pos = pos  # Position of the platform
        self.img = pg.image.load(r"images/still platform.png").convert()  # Load platform image
        self.img.set_colorkey((255, 255, 255), RLEACCEL)  # Set white as transparent
        self.height = self.img.get_height()  # Height of the platform
        self.width = self.img.get_width()  # Width of the platform
        self.type = type  # Type of the platform (e.g., "still" or "moving")
        self.vel = random.random()*3 + 1.5  # Velocity of the platform (for moving platforms)

    def display(self, surf):
        # Display the platform image on the specified surface
        surf.blit(self.img, self.pos)

    def collided_width(self, player):
        # Detect collisions between the platform and the player
        p1 = (self.pos[0] + self.width, self.pos[1] + self.height*0.1)
        p2 = (self.pos[0], self.pos[1] + self.height*0.1)

        if player.facing_right:
            p3 = (player.prev_pos[0] + 0.1*player.width, player.prev_pos[1] + player.height)
            p4 = (player.pos[0] + 0.1*player.width, player.pos[1] + player.height)
            p5 = (player.prev_pos[0] + 0.6*player.width, player.prev_pos[1] + player.height)
            p6 = (player.pos[0] + 0.6*player.width, player.pos[1] + player.height)
        else:
            p3 = (player.prev_pos[0] + 0.9*player.width, player.prev_pos[1] + player.height)
            p4 = (player.pos[0] + 0.9*player.width, player.pos[1] + player.height)
            p5 = (player.prev_pos[0] + 0.4*player.width, player.prev_pos[1] + player.height)
            p6 = (player.pos[0] + 0.4*player.width, player.pos[1] + player.height)

        return intersect(p1, p2, p5, p6) or intersect(p1, p2, p3, p4)

    def in_view_of(self, doodler, screen_height):
        # Check if the platform is within the view of the doodler
        return self.pos[1] < doodler.score_line + 0.66*screen_height + self.height

    def is_too_close_to(self, other_plat, screen_width):
        # Check if this platform is too close to another platform
        border_length = 1

        if self.type == "still" and other_plat.type == "still":
            l1p1 = (self.pos[0] - border_length, self.pos[1] - border_length)
            l1p2 = (self.pos[0] - border_length, self.pos[1] + border_length + self.height)
            l2p1 = (self.pos[0] + self.width + border_length, self.pos[1] - border_length)
            l2p2 = (self.pos[0] + self.width + border_length, self.pos[1] + border_length + self.height)

            l3p1 = (other_plat.pos[0] - border_length, other_plat.pos[1] - border_length)
            l3p2 = (other_plat.pos[0] + other_plat.width + border_length, other_plat.pos[1] - border_length)
            l4p1 = (other_plat.pos[0] - border_length, other_plat.pos[1] + border_length + other_plat.height)
            l4p2 = (other_plat.pos[0] + other_plat.width + border_length, other_plat.pos[1] + border_length + other_plat.height)

        else:
            l1p1 = (0, self.pos[1] - border_length)
            l1p2 = (screen_width, self.pos[1] - border_length)
            l2p1 = (0, self.pos[1] + border_length + self.height)
            l2p2 = (screen_width, self.pos[1] + border_length + self.height)

            l3p1 = (other_plat.pos[0] - border_length, other_plat.pos[1] - border_length)
            l3p2 = (other_plat.pos[0] - border_length, other_plat.pos[1] + border_length + other_plat.height)
            l4p1 = (other_plat.pos[0] + other_plat.width + border_length, other_plat.pos[1] - border_length)
            l4p2 = (other_plat.pos[0] + other_plat.width + border_length, other_plat.pos[1] + border_length + other_plat.height)

        return (intersect(l1p1, l1p2, l3p1, l3p2) or intersect(l1p1, l1p2, l4p1, l4p2) or intersect(l2p1, l2p2, l3p1, l3p2) or intersect(l2p1, l2p2, l4p1, l4p2))

    def move(self, screen_width):
        # Move the platform horizontally
        self.screen_wrap(screen_width)
        self.pos = (self.pos[0] + self.vel, self.pos[1])

    def screen_wrap(self, screen_width):
        # Wrap around the screen if the platform reaches the screen edges
        if self.pos[0] > screen_width - self.width:
            self.pos = (screen_width - self.width, self.pos[1])
            self.vel = -self.vel
        if self.pos[0] < 0:
            self.pos = (0, self.pos[1])
            self.vel = -self.vel

def intersect(p1, p2, p3, p4):
    # Helper function to detect intersection between two line segments
    return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)

def ccw(p1, p2, p3):
    # Helper function to determine if three points are in counter-clockwise order
    return (p3[1]-p1[1]) * (p2[0]-p1[0]) > (p2[1]-p1[1]) * (p3[0]-p1[0])

"""# main.py

This code represents a complete framework for evolving AI agents using NEAT to play the Jumpy game.

**Functions:**

* **get_best_genome_id**: This function iterates through a list of genomes and returns the ID of the genome with the highest fitness.

* **eval_genomes**: This function evaluates the fitness of each genome in a given population by simulating the game. It sets up the game environment, initializes the NEAT AI, runs the game simulation, and updates the fitness of each genome based on its performance.
run: This function initializes the NEAT algorithm with a configuration file, creates a population of genomes, and runs the NEAT algorithm by calling the eval_genomes function.

* **run** : This function initializes the NEAT algorithm, configures reporters for logging progress, and then runs the evolution process for a specified number of generations (50 in this case). Finally, it prints information about the winning genome, including its key and fitness.
"""

def get_best_genome_id(genomes):
    best_fitness = -1
    best_genome_id = None
    for genome_id, genome in genomes:
        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            best_genome_id = genome_id
    return best_genome_id


def eval_genomes(genomes, config):
    start_time = time.time()
    global gen_counter
    print(f"###################################-Generation {gen_counter}-#######################################################")
    gen_counter += 1
    # game setup portion from main with modification:
    pg.init()
    pg.display.init()
    WIDTH = 500
    HEIGHT = int(WIDTH * 1.75)

    # initialize the frame rate object and clock so that we can standardize frame rate
    clock = pg.time.Clock()
    FRAME_RATE = 60
    DT = (1200 / FRAME_RATE)

    # Setting up the drawing window
    screen = pg.display.set_mode([WIDTH, HEIGHT])
    pg.display.set_caption("Jumpy!")
    pg.font.init()
    my_font = pg.font.SysFont('Comic Sans MS', 25)

    # where to print the lines in the grid onscreen
    vertical_lines = [i for i in range(WIDTH) if i % 50 == 0]
    horizontal_lines = [i for i in range(HEIGHT) if i % 50 == 0]

    # store platform dimension variables
    temp_plat = Platform((0, 0), "still")
    plat_width = temp_plat.width
    plat_height = temp_plat.height

    # initialize object lists
    platforms = []
    platforms.append(Platform((WIDTH / 2, HEIGHT - 50), "still"))

    # generate initial platforms
    while len(platforms) <= 10:
        new_plat_pos = (random.random() * (WIDTH - plat_width), random.random() * (HEIGHT - 2 * plat_height) - plat_height - 40)
        new_plat = Platform(new_plat_pos, "still")
        is_too_close = False
        for platform in platforms:
            if new_plat.is_too_close_to(platform, WIDTH):
                is_too_close = True
        if not is_too_close:
            platforms.append(new_plat)

    dead_doodlers = []
    prev_score = 0
    best_doodler_score_keeper = 0

    # neat-AI setup
    networks = []  # for all doodlers' neural network
    ge = []  # list for neat-python's genomes
    doodlers = []  # each doodlers
    hitPlatforms = []  # list to holds the last highest hitting platform for each player

    for _ in range(len(genomes)):
        hitPlatforms.append(platforms[0])

    # number of iteration will base on POP_SIZE in the configuration file
    for genome_id, genome in genomes:
        genome.fitness = 0  # start with fitness level of 0
        network = neat.nn.FeedForwardNetwork.create(genome, config)  # create a network for the genome
        networks.append(network)
        ge.append(genome)

        doodler = Doodler((WIDTH / 2, 0.9 * HEIGHT))
        doodler.vel = (0, -0.025 * DT)
        doodler.score_line = 0.33 * HEIGHT
        doodlers.append(doodler)

    # Run until the user quits
    while True:
        # this will standardize speeds based on frame rate
        clock.tick(FRAME_RATE)

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()

        # Finding the best and worst doodlers based on their scores
        best_doodler = doodlers[0]  # assuming 1st one is best and worst
        worst_doodler = doodlers[0]
        for doodler in doodlers:
            if doodler.score > best_doodler.score:
                best_doodler = doodler
            if doodler.score < worst_doodler.score:
                worst_doodler = doodler

        # determine the chance a platform might be moving based on the doodler's score
        moving_chance = (best_doodler.score ** 0.5) / 100

        # create platforms that are always reachable by the doodler
        if int(best_doodler.score) % 240 == 0 and int(best_doodler.score) != prev_score:
            new_plat_type = "still"
            if random.random() < moving_chance:
                new_plat_type = "moving"
            platforms.append(Platform((random.random() * (WIDTH - plat_width), best_doodler.score - 30), new_plat_type))
            prev_score = int(best_doodler.score)

        # keep track of the number of platforms that will be put on the screen
        # this number will decrease as the player's score gets higher and eventually reach 0
        doodler_change = (worst_doodler.score_line + HEIGHT) / HEIGHT
        extra_platform_num = int((10 - (best_doodler.score / 2) ** 0.1) * doodler_change)

        # add the extra platforms to the screen where possible
        tries = 0

        while len(platforms) - 4 <= extra_platform_num:
            if tries > 10:
                break

            new_plat_pos = (random.random() * (WIDTH - plat_width), -1 * plat_height - 20 - random.random() * HEIGHT)
            new_plat_type = "still"
            if random.random() < 0.1:
                new_plat_type = "moving"
            new_plat = Platform(new_plat_pos, new_plat_type)
            is_too_close = False
            for platform in platforms:
                if new_plat.is_too_close_to(platform, WIDTH):
                    is_too_close = True
            if not is_too_close:
                platforms.append(Platform(new_plat_pos, new_plat_type))
                tries = 0
            tries += 1

        # scroll the game upwards as the best doodler gets higher and higher
        if best_doodler.pos[1] < 0.33 * HEIGHT and best_doodler.vel[1] < 0:
            offset = best_doodler.vel[1]
            for doodler in doodlers:
                if doodler != best_doodler:
                    doodler.pos = (doodler.pos[0], doodler.pos[1] - offset * DT)
                    doodler.score_line -= offset * DT
            best_doodler.score -= offset * DT
            best_doodler.pos = (best_doodler.pos[0], best_doodler.pos[1] - offset * DT)
            for platform in platforms:
                platform.pos = (platform.pos[0], platform.pos[1] - offset * DT)

        # control doodler scores and loss condition
        for doodler in doodlers:
            offset = doodler.vel[1]
            if doodler.pos[1] < doodler.score_line and offset < 0 and doodler != best_doodler:
                doodler.score -= offset * DT
                doodler.score_line += offset * DT

        # control doodler loss condition
        for doodler in doodlers:
            if doodler.is_dead(HEIGHT):
                dead_doodlers.append(doodler)
                doodlers.remove(doodler)

        if len(doodlers) == 0:
            pg.quit()
            break

        # remove platforms that are below the doodler with the lowest score
        for platform in platforms:
            if platform.pos[1] > worst_doodler.pos[1] + HEIGHT * 0.66 + plat_height:
                platforms.remove(platform)

        # control moving platform movement
        for platform in platforms:
            if platform.type == "moving":
                platform.move(WIDTH)

        # make the player be affected by gravity
        for doodler in doodlers:
            doodler.apply_gravity(DT)
            doodler.collision = False

        # if a doodler collides with a platform, it will bounce upwards
        for platform_id, platform in enumerate(platforms):
            for player_id, doodler in enumerate(doodlers):
                if platform.in_view_of(doodler, HEIGHT) and platform.collided_width(doodler) and doodler.vel[1] > 0:
                    doodler.collision = True
                    doodler.land_on_platform(DT, platform)

                    # reward player for hitting higher platform
                    if platform.pos[1] < hitPlatforms[player_id].pos[1]:
                        ge[player_id].fitness += 4
                        hitPlatforms[player_id] = platform

                    elif platform == hitPlatforms[player_id]:
                        # punish player for hitting same platform need to punish em hard to LEARN!
                        ge[player_id].fitness -= 2

        # move the player
        for doodler in doodlers:
            if not doodler.collision:
                doodler.move(DT)

        for platform in platforms:
            if platform.pos[1] > worst_doodler.pos[1] + HEIGHT * 0.66 + plat_height:
                platforms.remove(platform)

        # Fill the background with color and display the grid
        screen.fill((150, 123, 182))
        for y_pos in horizontal_lines:
            pg.draw.line(screen, (233, 225, 214), (0, y_pos), (WIDTH, y_pos))
        for x_pos in vertical_lines:
            pg.draw.line(screen, (233, 225, 214), (x_pos, 0), (x_pos, HEIGHT))

        # display all objects and doodlers
        for platform in platforms:
            platform.display(screen)

        for doodler in doodlers:
            if doodler.pos[1] < HEIGHT + doodler.height:
                doodler.display(screen)

        # display the best fitness , doodler's score , generation and alive count
        fit_list = []
        for g in ge:
            fit_list.append(g.fitness)

        best_fitness = my_font.render("Best Fitness: " + str(max(fit_list)), False, (0, 0, 0))
        screen.blit(best_fitness, (0.02 * WIDTH, 0.01 * HEIGHT))

        score = my_font.render("Best Score: " + str(int(best_doodler.score)), False, (0, 0, 0))
        screen.blit(score, (0.02 * WIDTH, 0.05 * HEIGHT))

        gen_text = my_font.render("Generation: " + str(gen_counter), False, (0, 0, 0))
        screen.blit(gen_text, (0.02 * WIDTH, 0.09 * HEIGHT))

        alive = my_font.render("Alive: " + str(len(doodlers)), False, (0, 0, 0))
        screen.blit(alive, (0.02 * WIDTH, 0.13 * HEIGHT))

        # update the screen
        pg.display.flip()


        for player_id, player in enumerate(doodlers):

            player_x, player_y = player.pos

            closest_platform_above_x = 0
            closest_platform_above_y = 0

            closest_platform_above_dist = float("inf")
            closest_platform_below_x, closest_platform_below_y = platforms[-1].pos  # ie , last platform
            closest_platform_below_dist = float("inf")

            for platform in platforms:
                platform_x, platform_y = platform.pos

                dist = (player_x - platform_x) ** 2 + (player_y - platform_y) ** 2

                # platform is above and is closer than current closest
                if platform_y < player_y and dist < closest_platform_above_dist:
                    # replace closest_platform_above
                    closest_platform_above_x = platform_x
                    closest_platform_above_y = platform_y
                    closest_platform_above_dist = dist

                # platform is below and is closer than current closest
                if platform_y > player_y and dist < closest_platform_below_dist:
                    # replace closest_platform_below
                    closest_platform_below_x = platform_x
                    closest_platform_below_y = platform_y
                    closest_platform_below_dist = dist

            # create a tuple of the values

            closest_platform_above = (closest_platform_above_x, closest_platform_above_y)
            closest_platform_below = (closest_platform_below_x, closest_platform_below_y)

            # display the input onscreen to see the learning process
            pg.draw.line(screen, (255, 0, 0), player.pos, closest_platform_above)
            pg.draw.line(screen, (255, 0, 0), player.pos, closest_platform_below)

            # calculate the relative distance between doodler and those platforms
            closest_platform_above_dist_x = player_x - closest_platform_above_x
            closest_platform_above_dist_y = player_y - closest_platform_above_y

            closest_platform_below_dist_x = player_x - closest_platform_below_x
            closest_platform_below_dist_y = closest_platform_below_y - player_y

            # feed the networks
            output = networks[player_id].activate((closest_platform_above_dist_x, closest_platform_above_dist_y,
                                                   closest_platform_below_dist_x, closest_platform_below_dist_y,
                                                   player.vel[1]))

            if output[0] > 0.5:
                player.update_movement(DT, True, False)
            if output[1] > 0.5:
                player.update_movement(DT, False, True)

        # draw out the red lines
        pg.display.flip()



        # update best_doodler_score_keeper every 5 sec
        if int(time.time() - start_time) % 5 == 0:
            best_doodler_score_keeper = int(best_doodler.score)

        # check every 10 sec if the best doodler survive without actually increasing the score
        # it probably doing back and forth movement so kill it

        if int(time.time() + 1 - start_time) % 10 == 0 and best_doodler_score_keeper == int(best_doodler.score):
            pg.quit()
            break



def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(generation_interval=5, time_interval_seconds=300))

    print("Let's Start!!!")
    global winner
    winner = population.run(eval_genomes, 30)

    print("\n\n\nWinner genome information:\n\n\n")
    print("Key:", winner.key)
    print("Fitness:", winner.fitness)
    print(winner)


if __name__ == '__main__':
    config_path = r"config-5.txt"
    global gen_counter
    gen_counter = 0
    run(config_path)

print(winner)