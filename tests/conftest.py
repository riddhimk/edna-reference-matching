import os

# The test suite must never upload its synthetic runs to the real cloud backup.
os.environ["EDNA_CLOUD_BACKUP"] = "off"
