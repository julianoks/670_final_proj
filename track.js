import { tracker } from "./algorithms/tracker.js"
import { transformBbox, invertTransformBbox, localBinaryPattern } from "./algorithms/utils.js"

/**
 * 
 * Example: new track().then(t => t.runInterval())
*/
export class track{
    constructor(parentEle = null, resolution = [300, 300]){
        this.is_ready = false;
        this.tracker = new tracker();
        this.resolution = resolution;
        this.parentEle = parentEle || document.body;
        // load model and video stream
        return cocoSsd.load().then(model => {
            this.model = model;
            return this.getVideoStream(this.resolution).then(video_stream => {
                this.video_stream = video_stream;
                this.is_ready = true;
                return this;
            });
        })

    }

    async getVideoStream(resolution = [300, 300]){
        const [width, height] = resolution;
        if(navigator.mediaDevices && navigator.mediaDevices.getUserMedia){
          return await navigator.mediaDevices.getUserMedia({
            audio: false, video: {width, height, facingMode: 'user'}
          }).then(stream => {
            let video = document.createElement('video');
            video.setAttribute('autoplay', true);
            video.srcObject = stream;
            video.getFrame = () => {
              const canvas = document.createElement('canvas');
              Object.assign(canvas, {width, height});
              canvas.getContext('2d').drawImage(video, 0, 0, width, height);
              return canvas;
            };
            return video;
          }).catch(e => {
              alert("Could not get video feed.");
              alert("Please accept video.");
            });
        } else {
            alert("Could not get video feed.");
            throw "COULDN'T GET `navigator.mediaDevices.getUserMedia`";
        }
      }

    /**
     * returns a promise resolving to a dict with keys
     *  img: an html image object of the frame
     *  matches: a dictionary with the matches from (TODO link)
     */
    async getNewFrameMatches(){
        const vidFrame = this.video_stream.getFrame();
        const img = tf.browser.fromPixels(vidFrame);
        const grayScale = img.mean(2);
        return Promise.all([
          this.model.detect(img),
          localBinaryPattern(grayScale)
        ]).then(([detections, lbpHist]) => {
          img.dispose();
          grayScale.dispose();
          let normalizedDetections = detections.map(d => new Promise((resolve) => resolve(
            Object.assign({}, d, {
              lbpFeature: lbpHist([d.bbox[0], d.bbox[1], d.bbox[2] + d.bbox[0], d.bbox[3] + d.bbox[1]]),
              bbox: transformBbox([d.bbox[0], d.bbox[1], d.bbox[2] + d.bbox[0], d.bbox[3] + d.bbox[1]], this.resolution)
            })
          )));
          return Promise.all(normalizedDetections);
        }).then(normalizedDetections => ({
            img: vidFrame,
            matches: this.tracker.matchDetections(normalizedDetections)
        }));
      }


    makeFrameDrawFn(){
        return () => {
            return this.getNewFrameMatches().then(({img, matches}) => {
                const ctx = img.getContext("2d");
                ctx.textAlign = "center";
                matches.tracking.forEach(to => {
                    const [tlx, tly, brx, bry] = invertTransformBbox(to.bbox, this.resolution);
                    ctx.beginPath();
                    const color = "#"+((1<<24)* (((12735 * to.id) % 2048))/(2048*12735)|0).toString(16);
                    ctx.strokeStyle = color;
                    ctx.fillStyle = color;
                    ctx.fillText(String(to.id) + " " + to.class, tlx, tly);
                    ctx.rect(tlx, tly, brx-tlx, bry-tly);
                    ctx.stroke();
                });
                while (this.parentEle.firstChild) { this.parentEle.firstChild.remove(); }
                this.parentEle.appendChild(img);
                return img;
            });
        }
    }

    runInterval(){
        this.interval = setInterval(this.makeFrameDrawFn(), 33);
        return this;
    }


}
