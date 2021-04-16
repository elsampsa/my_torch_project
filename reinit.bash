#!/bin/bash
name=$(basename $PWD)
if [ "$name" == "my_torch_project" ]
then
  echo will not overwrite default my_torch_project directory!
  exit
fi
echo initializing project $name
# rename python module directory
mv my_torch_project $name
# replace project names

fs="MANIFEST.in README.md setup.py"
for f in $fs
do
  find $f -exec sed -i -r "s/my_torch_project/$name/g" {} \;
  find $f -exec sed -i -r "s/your_package_name/$name/g" {} \;
done

find . -name "*.py" -exec sed -i -r "s/my_torch_project/$name/g" {} \;

# recompile documentation
#cd docs
#mkdir _static
#./clean.bash
# ./compile.bash # we can compile documentation only after this module is on the pythonpath!
cd ..
# decouple from git
# rm -rf .git .gitignore
rm -rf .git

if [ "$#" -eq 2 ]; then
    echo "Initializing to namespace "$1"."$2
    mkdir -p $1
    mv $name $1/$2
fi
