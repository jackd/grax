"""
Can be imported from `gin` to add directory and `grax/projects` to gin search path.

Example config file:

```gin
import grax.config
include "grax_config/single/fit.gin"
include "gat/configs/pubmed.gin"
```

"""
import os

import gin

base_dir = os.path.dirname(__file__)
for path in base_dir, os.path.join(base_dir, "projects"):
    gin.add_config_file_search_path(path)
