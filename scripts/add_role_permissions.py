#!/usr/bin/env python3
import requests
import sys


AIRFLOW_AUTH_URL = "http://airflow-webserver:8080/auth/fab/v1"
BASIC_AUTH = "YWlyZmxvdzphaXJmbG93"

PERMISSIONS = [
    ("Audit Logs", "can_read"),
    ("DAG Dependencies", "can_read"),
    ("DAG Code", "can_read"),
    ("DAG Runs", "can_read"),
    ("ImportError", "can_read"),
    ("Jobs", "can_read"),
    ("My Password", "can_read"),
    ("My Password", "can_edit"),
    ("My Profile", "can_read"),
    ("My Profile", "can_edit"),
    ("Plugins", "can_read"),
    ("SLA Misses", "can_read"),
    ("Task Instances", "can_read"),
    ("Task Logs", "can_read"),
    ("XComs", "can_read"),
    ("Website", "can_read"),
    ("Browse", "menu_access"),
    ("DAG Dependencies", "menu_access"),
    ("DAG Runs", "menu_access"),
    ("Documentation", "menu_access"),
    ("Docs", "menu_access"),
    ("Jobs", "menu_access"),
    ("Audit Logs", "menu_access"),
    ("Plugins", "menu_access"),
    ("SLA Misses", "menu_access"),
    ("Task Instances", "menu_access"),
    ("Task Instances", "can_create"),
    ("Task Instances", "can_edit"),
    ("Task Instances", "can_delete"),
    ("DAG Runs", "can_create"),
    ("DAG Runs", "can_edit"),
    ("DAG Runs", "can_delete"),
    ("Configurations", "can_read"),
    ("Admin", "menu_access"),
    ("Configurations", "menu_access"),
    ("Connections", "menu_access"),
    ("Pools", "menu_access"),
    ("Variables", "menu_access"),
    ("XComs", "menu_access"),
    ("Connections", "can_create"),
    ("Connections", "can_read"),
    ("Connections", "can_edit"),
    ("Connections", "can_delete"),
    ("Pools", "can_create"),
    ("Pools", "can_read"),
    ("Pools", "can_edit"),
    ("Pools", "can_delete"),
    ("Providers", "can_read"),
    ("Variables", "can_create"),
    ("Variables", "can_read"),
    ("Variables", "can_edit"),
    ("Variables", "can_delete"),
    ("XComs", "can_delete"),
    ("Task Reschedules", "can_read"),
    ("Task Reschedules", "menu_access"),
    ("Triggers", "can_read"),
    ("Triggers", "menu_access"),
    ("Roles", "can_read"),
    ("Users", "can_read"),
    ("Permissions", "can_read"),
    ("List Users", "menu_access"),
    ("Security", "menu_access"),
    ("List Roles", "menu_access"),
    ("User Stats Chart", "can_read"),
    ("User's Statistics", "menu_access"),
    ("View Menus", "can_read"),
    ("Permission Views", "can_read"),
    ("Providers", "menu_access"),
    ("XComs", "can_create")
]

ROLES = ["User"]


def create_permission_data():
    return [
        {
            "action": {"name": perm[1]},
            "resource": {"name": perm[0]}
        } for perm in PERMISSIONS
    ]


def update_role_permissions(role_name):
    try:
        data = {
            "actions": create_permission_data(),
            "name": role_name,
        }

        response = requests.patch(
            f"{AIRFLOW_AUTH_URL}/roles/{role_name}",
            json=data,
            headers={
                "accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Basic {BASIC_AUTH}"
            }
        )

        if response.status_code == 200:
            print(f"Successfully updated permissions for {role_name}")
            return True
        else:
            print(f"Failed to update {role_name}. Status: {response.status_code}", file=sys.stderr)
            print(f"Response: {response.text}", file=sys.stderr)
            print(f"{response}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"Error updating role {role_name}: {str(e)}", file=sys.stderr)
        return False


if __name__ == "__main__":
    for role in ROLES:
        print(f"Updating permissions for role: {role}")
        if not update_role_permissions(role):
            sys.exit(1)
