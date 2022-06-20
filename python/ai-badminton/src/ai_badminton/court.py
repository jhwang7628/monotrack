import numpy as np
import cv2
import matplotlib.pyplot as plt
import ipywidgets as wdg  # Using the ipython notebook widgets


'''
The court looks like this:
A-B-----C-----*-D
| |     |     | |
E-*-----*-----*-*
| |     |     | |
| |     |     | |
| |     |     | |
F---------------*
| |           | |
| |           | |
*---------------*
| |     |     | |
| |     |     | |
| |     |     | |
*-*-----*-----*-*
| |     |     | |
G-*-----*-----*-X

AC/AD = 1/2
AB/AD = 1.5/20
AE/AF = 2.5/15.5
AF/AG = 15.5/44
'''
class Court:
    eps = 0.05

    def __init__(self, corners = None):
        if corners is None:
            print('This court object must be manually labelled. See Court.manually_label.')
            return

        self.corners = corners

        # The corners must be in the order ADG (or some symmetry of it)
        npcorners = np.stack([np.array(c) for c in corners])
        lcorners = np.array([[0, 0], [1, 0], [0, 1], [1,1]])
        H, mask = cv2.findHomography(lcorners, npcorners, cv2.RANSAC, 2.0)
        self.H = H
        self.inv_H = np.linalg.inv(H)

        def to_coord(u, v):
            X = H @ np.array([u, v, 1])
            return X[:2] / X[2]

        # ratios of the shorter side
        self.sr = [0, 1.5/20, 1./2, 1-1.5/20, 1]
        self.lr = [0, 2.5/44, 15.5/44, 1/2, 1-15.5/44, 1-2.5/44, 1]
        self.points = []
        for v in self.lr:
            for u in self.sr:
                p = to_coord(u, v)
                self.points.append(p)

        self.lines = []
        # Draw the horizontals
        for v in self.lr:
            for i in range(1, len(self.sr)):
                up, u = self.sr[i-1], self.sr[i]
                p, c = to_coord(up, v), to_coord(u, v)
                self.lines.append((p, c))

        # Draw the verticals, but skip one line
        for u in self.sr:
            for i in range(1, len(self.lr)):
                vp, v = self.lr[i-1], self.lr[i]
                if (vp == 1./2 or v == 1./2) and u == 1./2:
                    continue
                p, c = to_coord(u, vp), to_coord(u, v)
                self.lines.append((p, c))

    def draw_lines(self, img):
        cimg = img.copy()
        colour = (0, 0, 255)
        thickness = 3
        for line in self.lines:
            p0, p1 = tuple(line[0].astype(int)), tuple(line[1].astype(int))
            cimg = cv2.line(cimg, p0, p1, colour, thickness)
        return cimg

    def pixel_to_court(self, p):
        x = self.inv_H @ np.array([p[0], p[1], 1])
        return x[:2] / x[2]

    def unnormalize_court_position(self, p):
        return np.array([p[0]*6.1, p[1]*13.41, 0])

    def in_court(self, p, slack=0):
        # 0 if not in court, 1 if in upper half, 2 if in lower half
        x = self.pixel_to_court(p)
        if not (-self.eps - slack[0] < x[0] < 1 + self.eps + slack[0]):
            return 0
        if not (-self.eps - slack[1] < x[1] < 1 + self.eps + slack[1]):
            return 0

        return 1 + (x[1] > 0.5 + self.eps)

    def draw_hit(self, img, pos, colour=(255,0,0)):
        # pos must be a vec in [0,1]^2 representing position on the 2d court
        centre = (int(pos[0] * img.shape[1]), int((1.-pos[1]) * img.shape[0]))
        radius = 4
        thickness = -1
        return cv2.circle(img, centre, radius, colour, thickness)

    '''
    A function for manually labelling the court points from an image.
        Input: An image with the courts
        Output: A `Court` object.
    '''
    def manually_label(self, frame):
        from matplotlib.backend_bases import MouseButton
        # Let's manually pick out the court pixel coordinates first
        fig = plt.figure()
        plt.imshow(frame)

        # Create and display textarea widget
        txt = wdg.Textarea(
            value='',
            placeholder='',
            description='event:',
            disabled=False
        )
        display(txt)

        # # Define a callback function that will update the textarea
        # frames = []
        # def onclick(event):
        #     txt.value = str(event)  # Dynamically update the text box above
        #     if event.button == 1:
        #         ret, frame = cap.read()
        #         frames.append(frame)
        #         plt.imshow(frame)
        #     elif event.button == 3:
        #         frames.pop()
        #         plt.imshow(frames[-1])

        # A basic UI function to store the court
        corners = []
        frames = [frame]
        def draw_dot(x, y):
            corners.append((x,y))
            centre = (int(x), int(y))
            radius = 5
            colour = (255, 0, 0)
            thickness = -1
            frame_next = cv2.circle(frames[-1].copy(), centre, radius, colour, thickness)
            frames.append(frame_next)
            plt.imshow(frames[-1])

        def undraw_dot():
            frames.pop()
            corners.pop()
            plt.imshow(frames[-1])

        def draw_court(corners):
            # Draws the court once we have the three corners
            court = Court(corners)
            frames[-1] = court.draw_lines(frames[-1])
            plt.imshow(frames[-1])

        def onclick(event):
            txt.value = str(event)  # Dynamically update the text box above
            if event.button == MouseButton.LEFT:
                x, y = event.xdata, event.ydata
                draw_dot(x, y)
                if len(corners) == 4:
                    draw_court(corners)
                    self.__init__(corners)
            elif event.button == MouseButton.RIGHT:
                if len(frames) > 1:
                    undraw_dot()

        # Create an hard reference to the callback not to be cleared by the garbage collector
        ka = fig.canvas.mpl_connect('button_press_event', onclick)

def read_court(filename):
    file = open(filename, 'r')
    coordinates = [[float(x) for x in line.split(';')] for line in file]
    return coordinates

def court_points_to_corners(pts):
    return [pts[1], pts[2], pts[0], pts[3]]

def court_points_to_corners_and_poles(pts):
    return court_points_to_corners(pts), [pts[4], pts[5]]
