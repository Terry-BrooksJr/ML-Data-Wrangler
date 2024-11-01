from invoke import task

@task(default=True)
def start_wrangler(ctx, port=8080):
    print(f"Starting Wrangler on port {port}...")
    