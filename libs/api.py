import requests


class UpdateStatus():
    def __init__(self, base_url, email, password, ):
        self.email = email
        self.password = password
        self.base_url = base_url
        self.login_url = f"{self.base_url}/api/v1/auth/email/login"
        self.allowed_status = ['Finished', 'Failed', 'Processing']
        
    def login(self):
        payload = {"email": self.email, "password": self.password}
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        resp = requests.post(self.login_url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json().get("token")

    def update_status(self, token, job_id, status):
        headers = {"accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        self.update_url = f"{self.base_url}/api/v1/VideoProcessingJobs/updateStatus/{job_id}"
        resp = requests.patch(self.update_url, headers=headers, data=f'"{status}"')
        resp.raise_for_status()
        return resp.json()

    def run(self, job_id, status):
        if status not in self.allowed_status:
            raise ValueError(f"Invalid status: {status}")
        token = self.login()
        return self.update_status(token, job_id, status)

# Example usage:
if __name__ == '__main__':
    base_url = "http://localhost:3001"
    email = "admin@example.com"
    password = "Adminpassword1@"
    updater = UpdateStatus(base_url, email, password)
    
    result = updater.run("01956c98-91e0-721c-a6f4-a4bb7658a39c", "Finished")
    print(result)