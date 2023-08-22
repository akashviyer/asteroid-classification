from flask import Flask, request, render_template
from src.pipeline.train_pipe import ModelEnsemble
from src.pipeline.predict_pipe import CustomData, PredictPipeline
app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/asteroid-classification', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        custom_data=CustomData(
            albedo=float(request.form.get('albedo')),
            diameter=float(request.form.get('diameter')),
            osc_semimaj_ax=float(request.form.get('osc_semimaj_ax')),
            osc_inclin=float(request.form.get('osc_inclin')),
            osc_eccentricity=float(request.form.get('osc_eccentricity'))
        )
    
    pred_df = custom_data.get_data_as_df()
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    return render_template('home.html', results=results[0])

application = app

if __name__ == '__main__':
    application.run(host="0.0.0.0", debug=True)
