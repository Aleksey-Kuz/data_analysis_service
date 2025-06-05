start:
	docker compose up --build

start-daemon:
	docker compose up --build -d

stop:
	docker compose down --volumes
