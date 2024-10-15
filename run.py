import uvicorn


def run():
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5321,
        reload=True,
        use_colors=True,
    )


if __name__ == "__main__":
    run()