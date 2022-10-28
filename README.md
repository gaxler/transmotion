# TransMotion

[Docs](https://gaxler.github.io/transmotion/doc/) | [Motion Transfer Example](https://gaxler.github.io/transmotion/book/demo.html)


The code that came with the
[FOMM](https://github.com/AliaksandrSiarohin/first-order-model) paper
became a bit of a standard in this area of research, but the code lack
annotation as to what kind of objects it passes around.

This lack of annotation and documentation makes it harder to understand
the code and scarier to tweak it.

This project is here to document what’s going on there. Key focus is on
type annotaoitn and use of `dataclasses` to be more explicit as to what
is being passed around and `einops` to be clearer about the shapes of
tensors.

#### This is still a work in progress. Most notably you’ll find missing

-   data loading functionality
-   training loop documentation


