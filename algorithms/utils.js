import {KalmanFilter} from './kalmanFilter.js';
import { hungarianOn3 } from './hungarianOn3.js';

/*** See https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
 * @param initVector the first vector in the sum
 * @param alpha the decay rate, 0 < alpha <= 1
 */
function exponentialMovingAverage(initVector, alpha){
    this.value = initVector;
    this.add = function(nextVector){
        for(let i=0; i<this.value.length; i++){
            this.value[i] *= 1 - alpha;
            this.value[i] += alpha * nextVector[i];
        }
    }
}

export function transformBbox(bbox, resolution){
    const [corner_x, corner_y, br_x, br_y] = bbox;
    const width = br_x - corner_x;
    const height = br_y - corner_y;
    const [w, h] = resolution;
    const centered = [(corner_x+(width/2))/w, (corner_y+(height/2))/h, (width/2)/w, (height/2)/h];
    return centered;
}

export function invertTransformBbox(bbox, resolution){
    const [centerx, centery, half_width, half_height] = bbox;
    const [w, h] = resolution;
    const corner_x = centerx - half_width;
    const corner_y = centery - half_height;
    const br_x = corner_x + (half_width*2);
    const br_y = corner_y + (half_height*2);
    return [corner_x * w, corner_y * h, br_x * w, br_y * h];
}

function bboxIOU(bbox1, bbox2){
    // intersection
    const x1 = Math.max(bbox1[0], bbox2[0]);
    const y1 = Math.max(bbox1[1], bbox2[1]);
    const x2 = Math.min(bbox1[2], bbox2[2]);
    const y2 = Math.min(bbox1[3], bbox2[3]);
    const w = x2 - x1;
    const h = y2 - y1;
    const inter = w<0 || h<0? 0 : w * h;
    // union
    let union = -1 * inter;
    union += (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
    union += (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
    return inter / union;
}

export function transformedIOU(tbbox1, tbbox2){
    return bboxIOU(invertTransformBbox(tbbox1, [1,1]), invertTransformBbox(tbbox2, [1,1]));
}



/**
 * 
 * @param {*} id
 * @param {*} initDetection An object with keys ("bbox", "class", "score", "feature")
 * @param {*} featureDecay the decay rate for the features, 0 < alpha <= 1, higher = more weight on recent feature vectors
 * @param {*} kalmanParams parameters to the Kalman Filter. See kalmanFilter.js. An object with keys ("initialCovariance", "transitionMatrix", "processNoise", "observationModel", "observationNoiseCovariance")
 */
export function KalmanTrackingObject(id, initDetection, featureDecay, kalmanParams){
    // the first 4 elements of the state are the center-normalized bbox, the next 2 are the (x,y) velocities of the center
    this.id = id;
    this.bbox = initDetection.bbox;
    const initState = initDetection.bbox.concat([0,0]);
    this.kf = new KalmanFilter(initState,
        kalmanParams.initialCovariance, kalmanParams.transitionMatrix,
        kalmanParams.processNoise);
    this.class = initDetection.class;
    this.iterationsSinceUpdate = 0;
    this.classFeature = new exponentialMovingAverage(initDetection.classFeature, featureDecay);
    this.lbpFeature = new exponentialMovingAverage(initDetection.lbpFeature, featureDecay);

    this.update = function(nextDetection){
        this.iterationsSinceUpdate = 0;
        this.class = nextDetection.class;
        this.classFeature.add(nextDetection.classFeature);
        this.bbox = nextDetection.bbox;
        const vx = nextDetection.bbox[0] - this.bbox[0];
        const vy = nextDetection.bbox[1] - this.bbox[1];
        const stateObservation = nextDetection.bbox.concat([vx, vy]);
        this.kf.update(stateObservation,
            kalmanParams.observationModel, kalmanParams.observationNoiseCovariance);
        return this;
    }

    this.updateNoObservation = function(){
        this.iterationsSinceUpdate += 1;
        this.bbox[0] += this.kf.state.elements[4];
        this.bbox[1] += this.kf.state.elements[5];
        const stateObservation = this.bbox.concat([this.kf.state.elements[4], this.kf.state.elements[5]]);
        this.kf.update(stateObservation,
            kalmanParams.observationModel, kalmanParams.observationNoiseCovariance);
        return this;
    }

}

/**
 * 
 * @param {tf.tensor2d} tfImg a grayscale image
 */
export async function localBinaryPattern(tfImg){
    let [n_rows, n_cols] = tfImg.shape;
    n_rows -= 2;
    n_cols -= 2;
    const img = tfImg.dataSync();
    let mat = [];
    for(let i=0; i < n_rows; i++){
        let row = [];
        for(let j=0; j < n_cols; j++){
            let val = 0;
            const center = img[((i+1) * (n_cols+2)) + j + 1];
            // consider parallelizing this?
            if(img[(i * (n_cols+2)) + j] > center){ val += 1; }
            if(img[(i * (n_cols+2)) + j + 1] > center){ val += 2; }
            if(img[(i * (n_cols+2)) + j + 2] > center){ val += 4; }
            if(img[((i+1) * (n_cols+2)) + j] > center){ val += 8; }
            if(img[((i+1) * (n_cols+2)) + j + 2] > center){ val += 16; }
            if(img[((i+2) * (n_cols+2)) + j] > center){ val += 32; }
            if(img[((i+2) * (n_cols+2)) + j + 1] > center){ val += 64; }
            if(img[((i+2) * (n_cols+2)) + j] + 2 > center){ val += 128; }
            row.push(val);
        }
        mat.push(row);
    }
    const makeHistogram = (bbox) => {
        let [x, y, x2, y2] = bbox;
        x2 = Math.min(n_cols, Math.floor(x2));
        y2 = Math.min(n_rows, Math.floor(y2));
        x = Math.max(0, Math.floor(x-1));
        y = Math.max(0, Math.floor(y-1));
        let hist = Array(256).fill(0);
        for(let i=y; i<y2; i++){
            for(let j=x; j<x2; j++){
                hist[mat[i][j]] += 1;
            }
        }
        hist = hist.map(x => x / ((1+y2-y) * (1+x2-x)));
        return hist;
    }
    return makeHistogram;
}


/**
 * Returns all birthed tracking objects that don't overlap more than `threshold` with every established tracking object.
 * @param {KalmanTrackingObject[]} birthsTOs 
 * @param {KalmanTrackingObject[]} establishedTOs 
 * @param {number} threshold
 */
export function suppressBirths(birthsTOs, establishedTOs, threshold){
    if(establishedTOs.length == 0){ return birthsTOs; }
    const established = establishedTOs.map(to => invertTransformBbox(to.bbox, [1,1]));
    return birthsTOs.filter(to => {
        const bbox = invertTransformBbox(to.bbox, [1,1]);
        return established.every(e => bboxIOU(e, bbox) <= threshold);
    });
}



export function bhattacharyya_coeff(probs1, probs2){
    let sum = 0;
    for(let i=0; i<probs1.length; i++){
        sum += Math.sqrt(probs1[i] * probs2[i]);
    }
    return sum;
}

/***
 * For every point in `prevPoints`, finds the indices of `k` nearest points in `nextPoints`.
 * These represent the candidate edges of the bipartite graph.
 * Returns a list of edges (v1, v2), where v1 and v2 are integer indices.
 * 
 * time performance using:
 * var [arr1, arr2] = [25, 100].map(size => Array(size).fill().map(()=>Array(10).fill().map(()=>Math.random())));
 *  console.time(); nearestNeighbors(arr1, arr2, 20); console.timeEnd();
 */
export function nearestNeighbors(prevPoints, nextPoints, k, maxDist){
    if(nextPoints.length == 0){ return Array(prevPoints.length).fill([]); }
    const dimensions = Array(nextPoints[0].length).fill().map((_,i) => i);
    const distance = (a1, a2) => Math.sqrt(dimensions.reduce((sum,i) => sum + Math.pow(a1[i] - a2[i], 2), 0));
    // HACK: because kdtree's knn doesn't return indices, append index, ignore it in dimensions
    for(let i in prevPoints){ prevPoints[i].push(i); }
    for(let i in nextPoints){ nextPoints[i].push(i); }
    const tree = new kdTree(nextPoints, distance, dimensions);
    const neighbors = prevPoints
        .flatMap((p,i) => tree.nearest(p, k, maxDist).map(([p,_]) => [i, +p[p.length-1]]));
    for(let i in prevPoints){ prevPoints[i].pop(); }
    for(let i in nextPoints){ nextPoints[i].pop(); }
    return neighbors;
}

/**
 * @param {*} edges an array of tuples (v1: int, v2: int, weight: float)
 */
export function greedyBipartiteSolver(edges){
	let left = new Set();
	let right = new Set();
	let selectedEdges = [];
	// sort by weight, in descending order
	edges.sort((e1, e2) => e1[2] < e2[2]? 1 : -1)
		.forEach(([v1, v2, _]) => {
			if(!left.has(v1) && !right.has(v2)){
				left.add(v1);
				right.add(v2);
				selectedEdges.push([v1, v2]);
			}
		});
	return selectedEdges;
}
 
/**
 * @param {*} edges an array of tuples (v1: int, v2: int, weight: float)
 */
export function hungarianBipartiteSolver(edges){
    const lefts = new Set(edges.map(([v1,v2,_]) => v1));
    const rights = new Set(edges.map(([v1,v2,_]) => v2));
    const size = Math.max(-1, ...lefts, ...rights);
    if(lefts.size == 0 || rights.size == 0){ return []; }
    const infeasible = Math.pow(2, 30);
    let matrix = Array(size + 1).fill().map(() => Array(size + 1).fill(infeasible));
    edges.forEach(([v1,v2,weight]) => {
        matrix[v1][v2] = -1 * weight;
    });
    const matches = hungarianOn3(matrix)
        .filter(([from, to]) => lefts.has(from) && rights.has(to) && matrix[from][to] != infeasible);
    return matches;
}
