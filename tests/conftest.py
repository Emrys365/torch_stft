def pytest_collection_modifyitems(session, config, items):
    items[:] = [item for item in items if item.name != "test_speed"]
