# SOVEREIGN LOGIC CORE - CONTRIBUTING GUIDELINES

Every change to this architecture must pass these four checks before commitment:

1. **Think Before Coding:** What exact problem is this solving? If there is no verifiable goal, do not write the code.
2. **Simplicity First:** Do not introduce complex abstractions (e.g., multi-agent swarms, theoretical math models) unless the simple baseline is proven to be insufficient.
3. **Surgical Changes:** Modify only what is necessary. Do not touch adjacent files or break existing interfaces (`scar_update`, etc.) without a documented architectural mandate.
4. **Goal-Driven Execution:** Every function must map to the physical reality of the Snapdragon 8 Elite hardware. Labels are not mechanisms. If it says it uses the NPU, it must contain the specific library calls to do so.
