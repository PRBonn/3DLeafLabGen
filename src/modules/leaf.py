import numpy as np 

MAX_CURL = 0.5

class Leaf():
    def __init__(self, id_:int, growth_stage: dict):
        # save parameters
        self.id = id_
        self.avg_length = growth_stage['l']
        self.avg_width = growth_stage['w']
        
        # generate values for leaf
        self.generate()

    def generate(self):
        self.generate_width()
        self.generate_length()
        self.generate_area()
        self.generate_curl()
        self.generate_angle()
    
    def generate_width(self):
        self.width = self.avg_width + np.random.choice((1,-1)) * (self.avg_width/4) * np.random.rand()
    
    def generate_length(self):
        self.length = self.avg_length + np.random.choice((1,-1)) * (self.avg_length/4) * np.random.rand()
 
    def generate_area(self):
        ellipse_area = self.compute_optimal_area()
        self.area = ellipse_area + np.random.choice((1,-1)) * ellipse_area / 8 * np.random.rand() 
 
    def compute_optimal_area(self):
        return np.pi * self.width * self.length

    def generate_curl(self):
        self.curl = np.round(MAX_CURL * np.random.rand() * 0.75,2)

    def generate_angle(self):
        self.angle = np.round(np.random.rand() * 0.75, 2)  # rads 

    def __str__(self):
        return f'This is leaf with id {self.id}. Width: {self.width} m. Length: {self.length} m. Area: {self.area} m^2. Curl: {self.curl} rad. Angle: {self.angle} rad.\n'
