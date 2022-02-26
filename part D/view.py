import matplotlib.pyplot as plt


class View:
    def __init__(self):
        self.curr_container = None
        self.prev_container = None
        self.rot_pts = None
        self.foe = None

    def set_params(self, curr_container, prev_container, rot_pts, foe):
        self.curr_container = curr_container
        self.prev_container = prev_container
        self.rot_pts = rot_pts
        self.foe = foe

    def show_result(self):
        fig, (curr_sec, prev_sec) = plt.subplots(1, 2, figsize=(12, 6))
        prev_sec.set_title('previous')
        prev_sec.imshow(self.prev_container.img)
        prev_p = self.prev_container.traffic_light
        prev_sec.plot(prev_p[:, 0], prev_p[:, 1], 'b+')

        curr_sec.set_title('current')
        curr_sec.imshow(self.curr_container.img)
        curr_p = self.curr_container.traffic_light
        curr_sec.plot(curr_p[:, 0], curr_p[:, 1], 'b+')

        for i in range(len(curr_p)):
            curr_sec.plot([curr_p[i, 0], self.foe[0]], [curr_p[i, 1], self.foe[1]], 'b')
            if self.curr_container.valid[i]:
                curr_sec.text(curr_p[i, 0], curr_p[i, 1],
                              r'{0:.1f}'.format(self.curr_container.traffic_lights_3d_location[i, 2]), color='r')
        curr_sec.plot(self.foe[0], self.foe[1], 'r+')
        curr_sec.plot(self.rot_pts[:, 0], self.rot_pts[:, 1], 'g+')
        plt.show()
