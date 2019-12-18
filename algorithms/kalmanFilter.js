/*
This function uses the Sylvester math library: http://sylvester.jcoglan.com/
The source for the Sylvester library can be found below,
https://cdnjs.cloudflare.com/ajax/libs/sylvester/0.1.3/sylvester.js
*/


export function KalmanFilter(
	initialState,      // vector of shape (d), required
	initialCovariance, // matrix of shape (d,d), defaults to identity
	transitionMatrix,  // matrix of shape (d,d), defaults to identity
	processNoise       // matrix of shape (d,d), defaults to the 0 matrix
	){
		this.d = initialState.length;
		this.age = 0;
		this.pdfCache = {};

		this.state = $V(initialState);
		this.covariance = $M(initialCovariance || Matrix.I(this.d));
		this.transitionMatrix = $M(transitionMatrix || Matrix.I(this.d));
		this.processNoise = $M(processNoise || Matrix.Zero(this.d, this.d));


		this.update = function(
			observation, // vector of shape (d), required
			observationModel, // matrix of shape (d,d), defaults to identity
			observationNoiseCovariance // matrix of shape (d,d), defaults to identity
			){
				// ingest observation
				observation = $V(observation);
				observationModel = $M(observationModel || Matrix.I(this.d));
				observationNoiseCovariance = $M(observationNoiseCovariance || Matrix.I(this.d));

				// predict next state and covariance
				const predState = this.transitionMatrix.multiply(this.state);
				const predCovariance = this.transitionMatrix
					.multiply(this.covariance)
					.multiply(this.transitionMatrix.transpose())
					.add(this.processNoise);
				
				// find gain
				const obsResidual = observation.subtract(observationModel.multiply(predState));
				const covResidual = observationModel
					.multiply(predCovariance)
					.multiply(observationModel.transpose())
					.add(observationNoiseCovariance);
				const optimalGain = predCovariance
					.multiply(observationModel.transpose())
					.multiply(covResidual.inverse());

				// update state and covariance
				const newState = predState.add(optimalGain.multiply(obsResidual));
				const newCovariance = Matrix.I(this.d)
					.subtract(optimalGain.multiply(observationModel))
					.multiply(predCovariance);
				this.state = newState;
				this.covariance = newCovariance;
				this.age += 1;
				return this;
		}


		// lazily compute and cache PDF values 
		this.getPDFCache = function(){
			if(this.pdfCache.age != this.age){
				// TODO: ensure nonzero, maybe compute psuedo det and inverse?
				const det = this.covariance.det();
				const denominator = Math.sqrt(det * Math.pow(2*Math.PI, this.d));
				const inverseCovariance = this.covariance.inverse();
				this.pdfCache = {age: this.age, denominator, inverseCovariance};
			}
			return this.pdfCache;
		}

		this.pdf = function(
			observation // vector of shape (d), required
			){
				observation = $V(observation);
				const {denominator, inverseCovariance} = this.getPDFCache();
				const centeredObs = $V(observation).subtract(this.state);
				const dot = centeredObs.dot(inverseCovariance.multiply(centeredObs));
				const numerator = Math.exp(-0.5 * dot);
				return numerator / denominator;
		}
}
