import unittest
import numpy as np
import app


class DashboardTests(unittest.TestCase):
    def test_marker_and_matrix_match_predictions_for_every_slider_value(self):
        for threshold in np.round(np.arange(.1, .901, .05), 2):
            with self.subTest(threshold=threshold):
                predictions = app.y_proba >= threshold
                positives = np.asarray(app.y_test) == 1
                expected_fpr = predictions[~positives].mean()
                expected_tpr = predictions[positives].mean()
                matrix_fig, roc_fig = app.update_model_graphs(float(threshold))
                self.assertAlmostEqual(roc_fig.data[2].x[0], expected_fpr)
                self.assertAlmostEqual(roc_fig.data[2].y[0], expected_tpr)
                expected = [[np.sum(~predictions & ~positives), np.sum(predictions & ~positives)],
                            [np.sum(~predictions & positives), np.sum(predictions & positives)]]
                np.testing.assert_array_equal(matrix_fig.data[0].z, expected)

    def test_dataset_and_split(self):
        self.assertEqual(len(app.df), 569)
        self.assertEqual(len(app.y_test), 143)
        self.assertFalse(set(app.X_train.index) & set(app.X_test.index))
        np.testing.assert_allclose(app.scaler.mean_, app.X_train.mean())

    def test_all_feature_figures(self):
        for feature in app.feature_cols:
            for figure in app.update_feature_graphs(feature):
                self.assertTrue(len(figure.data))
        self.assertTrue(len(app.build_heatmap().data))

    def test_http_routes(self):
        client = app.server.test_client()
        for route in ['/', '/_dash-layout', '/_dash-dependencies', '/assets/style.css']:
            with self.subTest(route=route):
                with client.get(route) as response:
                    self.assertEqual(response.status_code, 200)

    def test_actual_dash_callback(self):
        client = app.server.test_client()
        response = client.post('/_dash-update-component', json={
            'output': '..confusion-graph.figure...roc-graph.figure..',
            'outputs': [{'id': 'confusion-graph', 'property': 'figure'},
                        {'id': 'roc-graph', 'property': 'figure'}],
            'inputs': [{'id': 'threshold-slider', 'property': 'value', 'value': .1}],
            'state': [], 'changedPropIds': ['threshold-slider.value']})
        self.assertEqual(response.status_code, 200)
        marker = response.json['response']['roc-graph']['figure']['data'][2]
        self.assertAlmostEqual(marker['x'][0], (app.y_proba[np.asarray(app.y_test) == 0] >= .1).mean())
