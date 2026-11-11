
from locust import HttpUser, task, between

class BBCAUser(HttpUser):
    wait_time = between(1, 2)

    models = [
        "LinearRegression",
        "SVR",
        "RandomForest",
        "XGBoost",
        "LSTM"
    ]

    @task
    def predict_all_models(self):
        for model in self.models:
            with self.client.get(
                "/predict",
                params={"model": model},
                name=f"/predict [{model}]",
                catch_response=True
            ) as response:

                if response.status_code == 200:
                    try:
                        response.json()
                        response.success()
                    except ValueError:
                        response.failure("Response bukan JSON valid")
                else:
                    response.failure(
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
