import { KalmanTrackingObject, transformedIOU, suppressBirths, bhattacharyya_coeff, nearestNeighbors, greedyBipartiteSolver, hungarianBipartiteSolver } from "./utils.js";

export class tracker{
    constructor(){
        this.params = {
            matching: {
                minEdgeWeight: 0.1,
                birthScoreThreshold: 0.5,
                birthIOUThreshold: 0.3,
                k_candidate_matches: 10,
                neighborMaxDist: 1,
                featureDecay: 0.7,
                maxItersUnmatched: 5,
                useGreedySolver: false,
            },
            kalmanParams: {}, // TODO finetune kalmanParams, see https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html
            bipartiteWeights: {
                useIOU: true,
                useLBP: false,
                useClassFeature: true,
                useKalmanState: false
            },
        }

        this.id_counter = 0;
        this.trackingObjects = [];
    }


    getWeight(trackingObject, detection){
        const {classFeature, lbpFeature, bbox} = detection;
        let prob = 1;
        if(this.params.bipartiteWeights.useClassFeature){
            const probClassFeature = bhattacharyya_coeff(trackingObject.classFeature.value, classFeature);
            prob *= probClassFeature;
        }
        if(this.params.bipartiteWeights.useKalmanState){
            const vx = bbox[0] - trackingObject.bbox[0];
            const vy = bbox[1] - trackingObject.bbox[1];
            const probState = trackingObject.kf.pdf(bbox.concat([vx, vy]));
            prob *= Math.max(0.1, probState);
        }
        if(this.params.bipartiteWeights.useIOU){
            const probIOU = Math.max(0.1, transformedIOU(trackingObject.bbox, bbox));
            prob *= probIOU;
        }
        if(this.params.bipartiteWeights.useLBP){
            const probLBP = bhattacharyya_coeff(lbpFeature, trackingObject.lbpFeature.value);
            prob *= probLBP;
        }
        return prob;
    }


    /**
     * 
     * @param {*} trackingObjects 
     * @param {*} detections 
     * @param {*} k return the k nearest neighbors
     * @param {*} neighborMaxDist the maximum distance of a neighbor in nearest-neighbor search
     */
    makeBipartiteEdges(trackingObjects, detections, k, neighborMaxDist){
        const kalmanStates = trackingObjects.map(t => t.kf.state.elements);
        const detectedBboxes = detections.map(({bbox}) => bbox);
        const edges = nearestNeighbors(kalmanStates, detectedBboxes, k, neighborMaxDist)
            .filter(x => x.length != 0)
            .map(([v1, v2]) => [v1, v2, this.getWeight(trackingObjects[v1], detections[v2])]);
        return edges;
    }


    bipartiteSolver(edges){
        if(this.params.matching.useGreedySolver){
            return greedyBipartiteSolver(edges);
        } else {
            return hungarianBipartiteSolver(edges);
        }
    }


    /**
     * Associates detections to trackers
     * @param detections A list of dictionaries, with keys `bbox`, `lbpFeature`, `classFeature`, ...
     * @returns A dictionary with the keys `births`, `deaths`, and `tracking`, where
     *  `births` {KalmanTrackingObject[]} is a list of trackers that were created,
     *  `deaths` {KalmanTrackingObject[]} is a list of trackers that died, and
     *  `tracking` {KalmanTrackingObject[]} is a list of all active trackers 
     */
    matchDetections(detections){
        const {minEdgeWeight, birthScoreThreshold, birthIOUThreshold, k_candidate_matches, neighborMaxDist, featureDecay, maxItersUnmatched} = this.params.matching;

        const bipartiteEdges = this.makeBipartiteEdges(this.trackingObjects, detections, k_candidate_matches, neighborMaxDist)
            .filter(([v1,v2,weight]) => weight >= minEdgeWeight);
        const matches = this.bipartiteSolver(bipartiteEdges);
        const leftMatched = new Set(matches.map(m => m[0]));
        const rightMatched = new Set(matches.map(m => m[1]));
        const rightBirthed = Array(detections.length).fill().map((_,i) => i)
            .filter(i => !rightMatched.has(i))
            .filter(i => detections[i].score >= birthScoreThreshold);

        // update tracking objects that were matched
        matches.forEach(([v1, v2]) => this.trackingObjects[v1].update(detections[v2]));
        
        // generate new tracking objects for rightBirthed
        let births = rightBirthed.map(i => new KalmanTrackingObject(this.id_counter++, detections[i], featureDecay, this.params.kalmanParams));

        // populate list of tracking objects that died, and also those that weren't matched but are still alive
        const notMatched = this.trackingObjects.filter((_,i) => !leftMatched.has(i));
        const deaths = notMatched.filter(to => to.iterationsSinceUpdate >= maxItersUnmatched);
        const notMatchedStillAlive = notMatched.filter(to => to.iterationsSinceUpdate < maxItersUnmatched);
        // update tracking objects that were not matched but are still alive
        notMatchedStillAlive.forEach(to => to.updateNoObservation());

        const matchedTracking = this.trackingObjects.filter((_,i) => leftMatched.has(i));
        const nextTracking = matchedTracking.concat(notMatchedStillAlive);
        births = suppressBirths(births, nextTracking, birthIOUThreshold);
        this.trackingObjects = nextTracking.concat(births);
        return {births, deaths, tracking: this.trackingObjects};
    }
}

