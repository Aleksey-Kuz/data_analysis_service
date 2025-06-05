import requests
import time
from typing import Dict


AIRFLOW_AUTH_URL = "http://airflow-webserver:8080/auth/fab/v1"
BASIC_AUTH = "YWlyZmxvdzphaXJmbG93"

ROLES_USERS = {
    "User": {
        "first_name": "User",
        "last_name": "User",
        "email": "airflow_user@wa.com",
        "username": "airflow_user",
        "password": "airflow_user"
    }
}

class AirflowAPIClient:
    def __init__(self, base_url: str, base_auth: str, max_retries: int=20, initial_delay: float=1.):
        self.base_url = base_url
        self.base_auth = base_auth
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.session = requests.Session()
        self._authenticate()

    def _authenticate(self):
        self.session.headers.update({
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Basic {self.base_auth}",
        })

    def _request_with_retry(self, method, url, **kwargs):
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except (ConnectionError, requests.exceptions.RequestException) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.initial_delay * (2 * attempt)
                    print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
                    time.sleep(delay)
        raise last_exception

    def role_exists(self, role_name: str) -> bool:
        roles_url = f"{self.base_url}/roles"
        response = self._request_with_retry("GET", roles_url)
        roles = response.json().get("roles", [])
        return any(role["name"] == role_name for role in roles)

    def create_role(self, role_name: str) -> bool:
        if self.role_exists(role_name):
            print(f"Role {role_name} already exists")
            return True

        role_url = f"{self.base_url}/roles"
        response = self.session.post(role_url, json={
            "name": role_name,
            "actions": [
                {
                    "action": {"name": "can_read"},
                    "resource": {"name": "Audit Logs"}
                }
            ]
        })

        if response.status_code == 200:
            print(f"Successfully created role: {role_name}")
            return True
        else:
            print(f"Failed to create role {role_name}: {response.text}")
            print(f"{response}")
            return False

    def user_exists(self, username: str) -> bool:
        users_url = f"{self.base_url}/users"
        response = self.session.get(users_url)
        response.raise_for_status()
        users = response.json().get("users", [])
        return any(user["username"] == username for user in users)

    def create_user(self, user_data: Dict, role_name: str) -> bool:
        if self.user_exists(user_data["username"]):
            print(f"User {user_data['username']} already exists")
            return True

        user_url = f"{self.base_url}/users"
        response = self.session.post(user_url, json={
            "username": user_data["username"],
            "first_name": user_data["first_name"],
            "last_name": user_data["last_name"],
            "email": user_data["email"],
            "password": user_data["password"],
            "roles": [{"name": role_name}]
        })

        if response.status_code == 200:
            print(f"Successfully created user: {user_data['username']}")
            return True
        else:
            print(f"Failed to create user {user_data['username']}: {response.text}")
            print(f"{response}")
            return False


def main():
    client = AirflowAPIClient(AIRFLOW_AUTH_URL, BASIC_AUTH)

    print("Setting up roles...")
    for role_name in ROLES_USERS.keys():
        client.create_role(role_name)

    print("Setting up users...")
    for role_name, user_data in ROLES_USERS.items():
        client.create_user(user_data, role_name)


if __name__ == "__main__":
    main()
