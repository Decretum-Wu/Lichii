# Synthetic NUP fixture

`shifted_base_update.nup` was made with `nup pack` from
[a55uka/nup at a4840f0](https://github.com/a55uka/nup/tree/a4840f06c94ddeb414eea5b33c9704ffe995f11c).
It contains only synthetic ExeFS headers in `base/` and `updates/327680/`.
That version of the packer stores Build IDs under type `0x0006` instead of
`0x0007`. The base ID includes nine `FF` bytes so an unintended UTF-8
round trip expands the ID from 32 to 50 bytes.
