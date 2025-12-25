def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def compute_containment(bbox1, bbox2):
    px1, py1, px2, py2 = bbox1
    bx1, by1, bx2, by2 = bbox2
    
    intersection_x1 = max(px1, bx1)
    intersection_y1 = max(py1, by1)
    intersection_x2 = min(px2, bx2)
    intersection_y2 = min(py2, by2)
    
    if intersection_x2 < intersection_x1 or intersection_y2 < intersection_y1:
        return 0.0
        
    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
    bbox2_area = (bx2 - bx1) * (by2 - by1)
        
    return intersection_area / bbox2_area