# Test file to verify threejs template syntax
test_tpl = """
<!DOCTYPE html>
<html>
<head>
<title>Test</title>
</head>
<body>
<h1>__STATUS__</h1>
</body>
</html>
""".replace("__STATUS__", "REAL")
print("Template replace test passed:", "REAL" in test_tpl)
