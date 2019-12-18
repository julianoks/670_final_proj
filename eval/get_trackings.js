import { tracker } from "../algorithms/tracker.js"
import { transformBbox, invertTransformBbox, localBinaryPattern } from "../algorithms/utils.js"



/**
 * 
 * @param {Integer} nImages the number of images in the dataset
 * @param {String} datasetName 
 * @param {String} datasetTrainOrTest either 'train' or 'test'
 * @param {function} completionCallback A callback that recieves a list of dictionaries with keys 'id' (an int) and bbox (array of 4 numbers, [min_x, min_y, max_x, max_y])
 * 
 * Warning: this function was written as quickly as possible and probably belongs somewhere near the 4rd circle of callback hell. It could use some cleaning up, but hey, it works!
 */
export function getTrackings(nImages = 71, datasetName = 'TUD-Campus', datasetTrainOrTest = 'train', trackerWeightParams = {}, completionCallback = console.log){

    let allURLS = Array(nImages).fill().map((_,i)=>i+1).map(i => './2DMOT2015/' + datasetTrainOrTest + '/' + datasetName + '/img1/' + String(i).padStart(6, '0') + '.jpg');

    const firstImage = new Image();
    firstImage.src = allURLS[0];
    firstImage.onload = function(){
        const resolution = [firstImage.width, firstImage.height];
        runTracker(resolution, allURLS, completionCallback);
    }


    function runTracker(resolution, allURLS, completionCallback){
        cocoSsd.load().then(model => {
            const track = new tracker();
            Object.assign(track.params.bipartiteWeights, trackerWeightParams);

            async function runNextFrame(imageURL, callback){
                let img = new Image();
                img.src = imageURL;
                img.onload = function(){
                    img = tf.browser.fromPixels(img);
                    const grayScale = img.mean(2);
                    return Promise.all([
                        model.detect(img),
                        localBinaryPattern(grayScale)
                    ]).then(([detections, lbpHist]) => {
                        img.dispose();
                        grayScale.dispose();
                        let normalizedDetections = detections.map(d => new Promise((resolve) => resolve(
                            Object.assign({}, d, {
                                lbpFeature: lbpHist([d.bbox[0], d.bbox[1], d.bbox[2] + d.bbox[0], d.bbox[3] + d.bbox[1]]),
                                bbox: transformBbox([d.bbox[0], d.bbox[1], d.bbox[2] + d.bbox[0], d.bbox[3] + d.bbox[1]], resolution)
                              })
                        )));
                        normalizedDetections = Promise.all(normalizedDetections);
                        return normalizedDetections;
                    }).then(normalizedDetections => {
                    const matches = track.matchDetections(normalizedDetections);
                    callback(matches.tracking);
                    });
                }
            }

            let allTrackings = [];
            function customCallback(trackings){
                allTrackings.push(trackings.map(to => ({id: to.id, bbox: invertTransformBbox(to.bbox, resolution)})));
                if(allURLS.length == 0){
                    completionCallback(allTrackings);
                }
                else{
                    runNextFrame(allURLS.shift(), customCallback);
                }
            }

            runNextFrame(allURLS.shift(), customCallback);
        });
    }
}
