// Returns 1 if the lines intersect, otherwise 0. In addition, if the lines 
// intersect the intersection point may be stored in the floats i_x and i_y.
static void get_line_intersection(float p0_x, float p0_y, float p1_x, float p1_y, // first line segment
                                  float p2_x, float p2_y, float p3_x, float p3_y, // second line segment
                                  float *t_o) // output of distance along the first segment of intersection point
{
    float s1_x, s1_y, s2_x, s2_y;
    s1_x = p1_x - p0_x;     s1_y = p1_y - p0_y;
    s2_x = p3_x - p2_x;     s2_y = p3_y - p2_y;

    float denominator = (-s2_x * s1_y + s1_x * s2_y);

    // Lines are parallel
    if(denominator == 0)
        return;

    float s, t;
    s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / denominator;
    t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / denominator;

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1) {
        // Collision detected
        if (t_o != NULL)
            *t_o = t;
    }

    // No collision
    return;
}

// calculates the intersection of a line segment with a rectangle
static char get_rectangle_intersection(float p0_x, float p0_y, float p1_x, float p1_y, //line segment
                                       float* xs, float* ys,   // rectangle vertices
                                       float* t_o) // distance
{
    float ts[4] = {-1, -1, -1, -1};

    // loop over all edges of rectangle
    for(int i = 0; i<4; i++){
        get_line_intersection(p0_x, p0_y, p1_x, p1_y, xs[i % 4], ys[i % 4], xs[(i+1) % 4], ys[(i+1) % 4], &ts[i]);
    }

    // find smallest intersection point
    float min_t = 900;
    for(int i = 0; i<4; i++){
        if (ts[i] < min_t && ts[i] != -1)
            min_t = ts[i];
    }

    if (min_t < 900){
        *t_o = min_t;
        return 1;
    }

    return 0;
}

// rotate point around center for theta degree
static void  rotate(float x, float y, float theta, float center_x, float center_y, float *out_x, float *out_y){
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    *out_x = x * cos_theta - y * sin_theta + center_x;
    *out_y = x * sin_theta + y * cos_theta + center_y;

    return;
}

static void c_rect_corners(
    float center_x, float center_y,
    float length, float width, float angle,
    float *out_x, float *out_y)
{
    float half_l = length / 2;
    float half_w = width / 2;

    float x_pos[4] = {-half_l, -half_l, half_l, half_l};
    float y_pos[4] = {-half_w, half_w, half_w, -half_w};

    for(int i=0; i<4; i++){
        rotate(x_pos[i], y_pos[i], angle, center_x, center_y, &out_x[i], &out_y[i]);
    }

    return;
}

static void cmiddle_to_vertices(float x, float y, float length, float width, float angle, float* out){
    // convert the old represantation of the rectangle to vertices
    angle -= 1.5707; //pi/2

    float ux = width/2. * cos(angle);
    float uy = width/2. * sin(angle);
    float vx = -length/2. * sin(angle);
    float vy = length/2. * cos(angle);

    float ax = x - ux + vx;
    float ay = y - uy + vy;
    float bx = x + ux + vx;
    float by = y + uy + vy;
    float cx = x + ux - vx;
    float cy = y + uy - vy;
    float dx = x - ux - vx;
    float dy = y - uy - vy;

    out[0] = ax;
    out[1] = ay;
    out[2] = bx;
    out[3] = by;
    out[4] = cx;
    out[5] = cy;
    out[6] = dx;
    out[7] = dy;

    return;
}
static void cnormalize(float* v){
  /*
    Scale the vector to a length of 1
  */
    float x = v[0];
    float y = v[1];

    float norm = sqrt(x * x + y * y);
    v[0] = x/norm;
    v[1] = y/norm;
}


static float dot(float* a, float* b){
    /*
    Returns the dot (or scalar) product of the two vectors
    */
    return a[0] * b[0] + a[1] * b[1];
}

static int c_argmindist(double* l, double* x, double v[2], int N){
  int min_i = -1;
  double min_dist = 1e8;

  for(int i=0; i<2*N; i+=2){
    double dot_prod = x[i] * v[0] + x[i+1] * v[1];
    double dist = l[i/2] - 2 * dot_prod;

    if (dist <= min_dist){
      min_dist = dist;
      min_i = i/2;
    }
  }

  return min_i;
}


static double c_mindist(double* l, double* x, double v[2], int N){
  double min_dist = 1e8;

  for(int i=0; i<2*N; i+=2){
    double dot_prod = x[i] * v[0] + x[i+1] * v[1];
    double dist = l[i/2] - 2 * dot_prod;

    if (dist <= min_dist){
      min_dist = dist;
    }
  }

  return min_dist;
}

